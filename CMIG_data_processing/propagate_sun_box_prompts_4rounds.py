#!/usr/bin/env python3
"""Run four greedy SAM2 box-prompt rounds on SUN clips using 10 equally spaced candidate locations."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "CMIG_clips/SUN/sun_clips_train_stride_30_test_stride_100"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "CMIG_npz_data/sunseg"
DEFAULT_SAM2_ROOT = REPO_ROOT / "sam2"
DEFAULT_CHECKPOINT = DEFAULT_SAM2_ROOT / "checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DATASET_LABEL = "SUN"
EXPERIMENT_PREFIX = "sunseg"

NUM_CANDIDATES = 10
NUM_ROUNDS = 4
DEFAULT_BATCH_SIZE = 25
MASK_THRESHOLD = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-image-size", type=int, default=256)
    parser.add_argument("--memory-attention-layers", type=int, default=2)
    parser.add_argument("--initial-box-scale", type=float, default=1.4)
    parser.add_argument("--correction-box-scale", type=float, default=1.2)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--list-batches",
        action="store_true",
        help="List deterministic batch assignments without loading SAM2.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess videos whose four rounds are already complete.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail immediately when CUDA is unavailable instead of falling back to CPU.",
    )
    return parser.parse_args()


def scale_tag(value: float) -> str:
    scaled = value * 10
    if not math.isclose(scaled, round(scaled), abs_tol=1e-8):
        raise ValueError(
            f"Box scale {value} cannot be represented in the output name; "
            "use one decimal place."
        )
    return str(int(round(scaled)))


def experiment_name(initial_scale: float, correction_scale: float, batch_index: int) -> str:
    return (
        f"{EXPERIMENT_PREFIX}_{scale_tag(initial_scale)}_{scale_tag(correction_scale)}_"
        f"batch_{batch_index}"
    )


def list_video_names(images_root: Path, masks_root: Path) -> list[str]:
    if not images_root.is_dir():
        raise FileNotFoundError(f"{DATASET_LABEL} clip image root does not exist: {images_root}")
    if not masks_root.is_dir():
        raise FileNotFoundError(f"{DATASET_LABEL} clip mask root does not exist: {masks_root}")
    image_names = {path.name for path in images_root.iterdir() if path.is_dir()}
    mask_names = {path.name for path in masks_root.iterdir() if path.is_dir()}
    if image_names != mask_names:
        raise RuntimeError(
            f"Image/mask clip folders differ: images-only={len(image_names-mask_names)}, "
            f"masks-only={len(mask_names-image_names)}"
        )
    return sorted(image_names)


def make_batches(video_names: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    return [
        video_names[index : index + batch_size]
        for index in range(0, len(video_names), batch_size)
    ]


def sampled_frame_indices(video_length: int) -> list[int]:
    if video_length < NUM_CANDIDATES:
        raise ValueError(
            f"A video needs at least {NUM_CANDIDATES} frames; got {video_length}"
        )
    indices = np.linspace(0, video_length - 1, NUM_CANDIDATES, dtype=int).tolist()
    if len(set(indices)) != NUM_CANDIDATES:
        raise RuntimeError(f"Equal sampling did not produce {NUM_CANDIDATES} unique frames")
    return indices


def load_binary_masks(mask_dir: Path, frame_stems: list[str]) -> tuple[list[np.ndarray], list[int] | None]:
    masks: list[np.ndarray] = []
    palette = None
    shape = None
    for stem in frame_stems:
        path = mask_dir / f"{stem}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {DATASET_LABEL} mask: {path}")
        with Image.open(path) as image:
            if palette is None:
                palette = image.getpalette()
            mask = np.asarray(image)
        if mask.ndim == 3:
            mask = np.any(mask > 0, axis=2)
        else:
            mask = mask > 0
        mask = mask.astype(np.uint8)
        if shape is None:
            shape = mask.shape
        elif mask.shape != shape:
            raise RuntimeError(f"Mask shape changed within {mask_dir}: {shape} to {mask.shape}")
        if not np.any(mask):
            raise RuntimeError(f"{DATASET_LABEL} clip contains an empty saved mask: {path}")
        masks.append(mask)
    return masks, palette


def scaled_box(mask: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Box scales must be greater than zero")
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Cannot construct a box around an empty mask")
    height, width = mask.shape
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    box_width = xmax - xmin + 1.0
    box_height = ymax - ymin + 1.0
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    half_width = box_width * scale / 2.0
    half_height = box_height * scale / 2.0
    return np.array(
        [
            max(0.0, center_x - half_width),
            max(0.0, center_y - half_height),
            min(float(width - 1), center_x + half_width),
            min(float(height - 1), center_y + half_height),
        ],
        dtype=np.float32,
    )


def autocast_context(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def print_cuda_diagnostics() -> bool:
    """Print the runtime details needed to diagnose CUDA availability in SLURM."""
    print("\nCUDA/PyTorch diagnostics", flush=True)
    print(f"Python executable: {sys.executable}", flush=True)
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"PyTorch compiled CUDA runtime: {torch.version.cuda}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '<unset>')}", flush=True)
    try:
        cuda_available = torch.cuda.is_available()
        print(f"torch.cuda.is_available(): {cuda_available}", flush=True)
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}", flush=True)
        if cuda_available:
            device_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_index)
            print(f"Current CUDA device index: {device_index}", flush=True)
            print(f"GPU name: {properties.name}", flush=True)
            print(f"GPU compute capability: {properties.major}.{properties.minor}", flush=True)
            print(
                f"GPU total memory GiB: {properties.total_memory / (1024**3):.2f}",
                flush=True,
            )
            print(f"cuDNN version: {torch.backends.cudnn.version()}", flush=True)
    except Exception as error:
        print(f"CUDA diagnostic exception: {type(error).__name__}: {error}", flush=True)
        cuda_available = False
    print("End CUDA/PyTorch diagnostics\n", flush=True)
    return cuda_available


def propagate_with_boxes(
    predictor,
    inference_state,
    selected_labels: list[int],
    label_to_frame: dict[int, int],
    boxes: dict[int, np.ndarray],
    device: str,
) -> dict[int, np.ndarray]:
    """Reset tracking, apply every selected box, and return binary masks for all frames."""
    with autocast_context(device):
        predictor.reset_state(inference_state)
        for label in selected_labels:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=label_to_frame[label],
                obj_id=1,
                box=boxes[label],
                clear_old_points=True,
                normalize_coords=True,
            )

        segments: dict[int, np.ndarray] = {}
        for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(inference_state):
            combined = np.zeros(mask_logits.shape[-2:], dtype=bool)
            for object_index, _object_id in enumerate(object_ids):
                combined |= mask_logits[object_index].detach().cpu().numpy().squeeze() > MASK_THRESHOLD
            segments[int(frame_idx)] = combined
    return segments


def per_frame_ious(predictions: dict[int, np.ndarray], gt_masks: list[np.ndarray]) -> list[float]:
    scores: list[float] = []
    for frame_idx, gt_mask in enumerate(gt_masks):
        prediction = predictions.get(frame_idx)
        if prediction is None:
            prediction = np.zeros_like(gt_mask, dtype=bool)
        else:
            prediction = prediction.astype(bool)
        ground_truth = gt_mask.astype(bool)
        intersection = np.logical_and(prediction, ground_truth).sum()
        union = np.logical_or(prediction, ground_truth).sum()
        scores.append(1.0 if union == 0 else float(intersection / union))
    return scores


def json_key(fixed_labels: list[int], candidate: int | None = None) -> str:
    structure = [fixed_labels] if candidate is None else [fixed_labels, [candidate]]
    return json.dumps(structure, separators=(",", ":"))


def save_masks_atomic(
    output_dir: Path,
    predictions: dict[int, np.ndarray],
    frame_stems: list[str],
    palette: list[int] | None,
) -> None:
    temporary_dir = output_dir.parent / f".{output_dir.name}.tmp"
    shutil.rmtree(temporary_dir, ignore_errors=True)
    temporary_dir.mkdir(parents=True)
    try:
        for frame_idx, stem in enumerate(frame_stems):
            mask = predictions.get(frame_idx)
            if mask is None:
                raise RuntimeError(f"SAM2 did not return frame {frame_idx}")
            image = Image.fromarray(mask.astype(np.uint8), mode="P")
            if palette is not None:
                image.putpalette(palette)
            image.save(temporary_dir / f"{stem}.png")
        shutil.rmtree(output_dir, ignore_errors=True)
        temporary_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def save_json_atomic(path: Path, payload: dict[str, list[float]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    temporary_path.replace(path)


def video_is_complete(output_dir: Path, video_name: str, frame_count: int) -> bool:
    masks_root = output_dir / "sam2_masks"
    info_root = output_dir / "info_dict"
    for round_number in range(1, NUM_ROUNDS + 1):
        mask_dir = masks_root / f"{video_name}_round_{round_number}"
        json_path = info_root / f"{video_name}_round_{round_number}.json"
        if not json_path.is_file() or not mask_dir.is_dir():
            return False
        if sum(path.is_file() and path.suffix.lower() == ".png" for path in mask_dir.iterdir()) != frame_count:
            return False
    return True


def remove_partial_video_outputs(output_dir: Path, video_name: str) -> None:
    for round_number in range(1, NUM_ROUNDS + 1):
        shutil.rmtree(
            output_dir / "sam2_masks" / f"{video_name}_round_{round_number}",
            ignore_errors=True,
        )
        json_path = output_dir / "info_dict" / f"{video_name}_round_{round_number}.json"
        json_path.unlink(missing_ok=True)


def load_reduced_memory_checkpoint(
    predictor, checkpoint_path: Path, memory_attention_layers: int
) -> None:
    """Load a SAM2 checkpoint, allowing only removed memory-attention layers to be discarded."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
    model_state = predictor.state_dict()
    shape_mismatches = [
        key
        for key, value in checkpoint.items()
        if key in model_state and value.shape != model_state[key].shape
    ]
    if shape_mismatches:
        raise RuntimeError(f"Checkpoint tensor-shape mismatches: {shape_mismatches}")

    discarded = [key for key in checkpoint if key not in model_state]
    allowed_prefixes = tuple(
        f"memory_attention.layers.{index}."
        for index in range(memory_attention_layers, 100)
    )
    invalid_discarded = [key for key in discarded if not key.startswith(allowed_prefixes)]
    if invalid_discarded:
        raise RuntimeError(
            "Checkpoint has unexpected tensors unrelated to removed memory-attention "
            f"layers: {invalid_discarded}"
        )

    compatible = {key: value for key, value in checkpoint.items() if key in model_state}
    missing, unexpected = predictor.load_state_dict(compatible, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load was incomplete: missing={missing}, unexpected={unexpected}"
        )
    print(
        f"Loaded {len(compatible)} checkpoint tensors; discarded {len(discarded)} "
        f"tensor(s) from memory-attention layers >= {memory_attention_layers}."
    )


def process_video(
    predictor,
    video_name: str,
    images_root: Path,
    masks_root: Path,
    output_dir: Path,
    initial_scale: float,
    correction_scale: float,
    device: str,
    overwrite: bool,
) -> None:
    image_dir = images_root / video_name
    mask_dir = masks_root / video_name
    frame_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if not frame_paths:
        raise RuntimeError(f"No JPEG frames found for {video_name}")
    frame_stems = [path.stem for path in frame_paths]

    if video_is_complete(output_dir, video_name, len(frame_stems)) and not overwrite:
        print(f"Skipping complete video: {video_name}")
        return
    remove_partial_video_outputs(output_dir, video_name)

    gt_masks, palette = load_binary_masks(mask_dir, frame_stems)
    sampled_indices = sampled_frame_indices(len(frame_stems))
    label_to_frame = dict(enumerate(sampled_indices))
    boxes = {
        label: scaled_box(
            gt_masks[frame_idx],
            initial_scale if label == 0 else correction_scale,
        )
        for label, frame_idx in label_to_frame.items()
    }
    print(f"{video_name}: label-to-frame map {label_to_frame}")

    with autocast_context(device):
        inference_state = predictor.init_state(video_path=str(image_dir))

    selected_labels = [0]
    selection_log = []
    started = time.time()
    for round_number in range(1, NUM_ROUNDS + 1):
        print(f"  Round {round_number}: fixed labels {selected_labels}")
        baseline_predictions = propagate_with_boxes(
            predictor, inference_state, selected_labels, label_to_frame, boxes, device
        )
        baseline_ious = per_frame_ious(baseline_predictions, gt_masks)
        round_info: dict[str, list[float]] = {
            json_key(selected_labels): baseline_ious
        }

        save_masks_atomic(
            output_dir / "sam2_masks" / f"{video_name}_round_{round_number}",
            baseline_predictions,
            frame_stems,
            palette,
        )

        remaining_labels = [
            label for label in range(1, NUM_CANDIDATES) if label not in selected_labels
        ]
        candidate_means: dict[int, float] = {}
        for candidate in remaining_labels:
            trial_labels = selected_labels + [candidate]
            trial_predictions = propagate_with_boxes(
                predictor, inference_state, trial_labels, label_to_frame, boxes, device
            )
            trial_ious = per_frame_ious(trial_predictions, gt_masks)
            round_info[json_key(selected_labels, candidate)] = trial_ious
            candidate_means[candidate] = float(np.mean(trial_ious))
            print(
                f"    candidate label={candidate}, frame={label_to_frame[candidate]}, "
                f"mean IoU={candidate_means[candidate]:.6f}"
            )

        save_json_atomic(
            output_dir / "info_dict" / f"{video_name}_round_{round_number}.json",
            round_info,
        )

        best_candidate = max(candidate_means, key=lambda label: (candidate_means[label], -label))
        selection_log.append(
            {
                "round": round_number,
                "fixed_labels": selected_labels.copy(),
                "fixed_frames": [label_to_frame[label] for label in selected_labels],
                "baseline_mean_iou": float(np.mean(baseline_ious)),
                "best_candidate_label": best_candidate,
                "best_candidate_frame": label_to_frame[best_candidate],
                "best_candidate_mean_iou": candidate_means[best_candidate],
            }
        )
        if round_number < NUM_ROUNDS:
            selected_labels.append(best_candidate)

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_payload = {
        "video": video_name,
        "label_to_frame": label_to_frame,
        "initial_box_scale": initial_scale,
        "correction_box_scale": correction_scale,
        "rounds": selection_log,
        "elapsed_seconds": time.time() - started,
    }
    save_json_atomic(log_dir / f"{video_name}.json", log_payload)
    print(f"Completed {video_name} in {log_payload['elapsed_seconds']:.2f} seconds")


def main() -> None:
    args = parse_args()
    if args.initial_box_scale <= 0 or args.correction_box_scale <= 0:
        raise ValueError("Box scales must be greater than zero")

    images_root = args.dataset_root / "Images"
    masks_root = args.dataset_root / "Masks"
    all_video_names = list_video_names(images_root, masks_root)
    batches = make_batches(all_video_names, args.batch_size)

    if args.list_batches:
        for index, batch in enumerate(batches):
            first = batch[0] if batch else "-"
            last = batch[-1] if batch else "-"
            print(f"batch {index}: clips={len(batch)}, first={first}, last={last}")
        return
    if args.batch_index < 0 or args.batch_index >= len(batches):
        raise ValueError(
            f"--batch-index must be between 0 and {len(batches)-1}; got {args.batch_index}"
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {args.checkpoint}")

    selected_videos = batches[args.batch_index]
    output_dir = args.output_root / experiment_name(
        args.initial_box_scale, args.correction_box_scale, args.batch_index
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Total clips: {len(all_video_names)}")
    print(f"Batch {args.batch_index}/{len(batches)-1}: {len(selected_videos)} clips")
    print(f"Output: {output_dir}")

    sam2_root = str(args.sam2_root.resolve())
    if sam2_root not in os.sys.path:
        os.sys.path.insert(0, sam2_root)
    from sam2.build_sam import build_sam2_video_predictor

    cuda_available = print_cuda_diagnostics()
    if args.require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA is unavailable. Refusing CPU fallback because --require-cuda was set."
        )
    device = "cuda" if cuda_available else "cpu"
    print(f"Selected inference device: {device}", flush=True)
    predictor = build_sam2_video_predictor(
        config_file=args.model_config,
        ckpt_path=None,
        device=device,
        hydra_overrides_extra=[
            f"++model.image_size={args.model_image_size}",
            f"++model.memory_attention.num_layers={args.memory_attention_layers}",
        ],
    )
    load_reduced_memory_checkpoint(
        predictor, args.checkpoint, args.memory_attention_layers
    )

    for position, video_name in enumerate(selected_videos, start=1):
        print(f"\n[{position}/{len(selected_videos)}] {video_name}")
        process_video(
            predictor=predictor,
            video_name=video_name,
            images_root=images_root,
            masks_root=masks_root,
            output_dir=output_dir,
            initial_scale=args.initial_box_scale,
            correction_scale=args.correction_box_scale,
            device=device,
            overwrite=args.overwrite,
        )

    print("Batch complete.")


if __name__ == "__main__":
    main()
