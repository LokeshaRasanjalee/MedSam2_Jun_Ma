#!/usr/bin/env python3
"""Run four greedy SAM2 mask-prompt rounds on MUP clips: frame 0 uses a non-expert mask and every correction uses an expert mask."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import propagate_sun_box_prompts_4rounds as common

common.DATASET_LABEL = "MUP"


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "CMIG_clips/MUP/mup_clips_train_stride_5_val_test_stride_10"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "CMIG_npz_data/mup"
DEFAULT_SAM2_ROOT = REPO_ROOT / "sam2"
DEFAULT_CHECKPOINT = DEFAULT_SAM2_ROOT / "checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

NUM_FRAMES = 10
NUM_ROUNDS = 4
DEFAULT_BATCH_SIZE = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-image-size", type=int, default=256)
    parser.add_argument("--memory-attention-layers", type=int, default=2)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--list-batches",
        action="store_true",
        help="List deterministic clip batches without loading SAM2.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess clips whose four output rounds are already complete.",
    )
    return parser.parse_args()


def list_clip_names(
    scans_root: Path, non_expert_root: Path, expert_root: Path
) -> list[str]:
    roots = {
        "micro_ultrasound_scans": scans_root,
        "non_expert_annotations": non_expert_root,
        "expert_annotations": expert_root,
    }
    names_by_root: dict[str, set[str]] = {}
    for label, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"MUP {label} root does not exist: {root}")
        names_by_root[label] = {path.name for path in root.iterdir() if path.is_dir()}

    reference = names_by_root["micro_ultrasound_scans"]
    for label, names in names_by_root.items():
        if names != reference:
            raise RuntimeError(
                f"MUP clip folders differ for {label}: "
                f"missing={len(reference - names)}, extra={len(names - reference)}"
            )
    return sorted(reference)


def make_batches(clip_names: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    return [
        clip_names[index : index + batch_size]
        for index in range(0, len(clip_names), batch_size)
    ]


def propagate_with_masks(
    predictor,
    inference_state,
    selected_frames: list[int],
    prompt_masks: dict[int, np.ndarray],
    device: str,
) -> dict[int, np.ndarray]:
    """Reset tracking, add all selected mask prompts, and propagate over the clip."""
    with common.autocast_context(device):
        predictor.reset_state(inference_state)
        for frame_index in selected_frames:
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_index,
                obj_id=1,
                mask=prompt_masks[frame_index],
            )

        predictions: dict[int, np.ndarray] = {}
        for frame_index, object_ids, mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            combined = np.zeros(mask_logits.shape[-2:], dtype=bool)
            for object_index, _object_id in enumerate(object_ids):
                combined |= (
                    mask_logits[object_index].detach().cpu().numpy().squeeze()
                    > common.MASK_THRESHOLD
                )
            predictions[int(frame_index)] = combined
    return predictions


def process_clip(
    predictor,
    clip_name: str,
    scans_root: Path,
    non_expert_root: Path,
    expert_root: Path,
    output_dir: Path,
    device: str,
    overwrite: bool,
) -> None:
    scan_dir = scans_root / clip_name
    frame_paths = sorted(
        path
        for path in scan_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if len(frame_paths) != NUM_FRAMES:
        raise RuntimeError(
            f"MUP clip {clip_name} must contain {NUM_FRAMES} JPEG frames; "
            f"found {len(frame_paths)}"
        )
    frame_stems = [path.stem for path in frame_paths]

    if common.video_is_complete(output_dir, clip_name, NUM_FRAMES) and not overwrite:
        print(f"Skipping complete clip: {clip_name}")
        return
    common.remove_partial_video_outputs(output_dir, clip_name)

    expert_masks, expert_palette = common.load_binary_masks(
        expert_root / clip_name, frame_stems
    )
    non_expert_masks, _ = common.load_binary_masks(
        non_expert_root / clip_name, frame_stems
    )

    # Frame 0 is always prompted with its non-expert annotation. Every other
    # candidate/fixed correction is prompted with its expert annotation.
    prompt_masks = {0: non_expert_masks[0]}
    prompt_masks.update({index: expert_masks[index] for index in range(1, NUM_FRAMES)})

    with common.autocast_context(device):
        inference_state = predictor.init_state(video_path=str(scan_dir))

    selected_frames = [0]
    selection_log: list[dict[str, object]] = []
    started = time.time()

    for round_number in range(1, NUM_ROUNDS + 1):
        print(f"  Round {round_number}: fixed frames {selected_frames}")
        baseline_predictions = propagate_with_masks(
            predictor, inference_state, selected_frames, prompt_masks, device
        )
        baseline_ious = common.per_frame_ious(baseline_predictions, expert_masks)
        round_info: dict[str, list[float]] = {
            common.json_key(selected_frames): baseline_ious
        }

        common.save_masks_atomic(
            output_dir / "sam2_masks" / f"{clip_name}_round_{round_number}",
            baseline_predictions,
            frame_stems,
            expert_palette,
        )

        remaining_frames = [
            index for index in range(1, NUM_FRAMES) if index not in selected_frames
        ]
        candidate_means: dict[int, float] = {}
        for candidate in remaining_frames:
            trial_predictions = propagate_with_masks(
                predictor,
                inference_state,
                selected_frames + [candidate],
                prompt_masks,
                device,
            )
            trial_ious = common.per_frame_ious(trial_predictions, expert_masks)
            round_info[common.json_key(selected_frames, candidate)] = trial_ious
            candidate_means[candidate] = float(np.mean(trial_ious))
            print(
                f"    expert correction frame={candidate}, "
                f"mean IoU={candidate_means[candidate]:.6f}"
            )

        common.save_json_atomic(
            output_dir / "info_dict" / f"{clip_name}_round_{round_number}.json",
            round_info,
        )

        best_candidate = max(
            candidate_means, key=lambda index: (candidate_means[index], -index)
        )
        selection_log.append(
            {
                "round": round_number,
                "fixed_frames": selected_frames.copy(),
                "fixed_prompt_sources": [
                    "non_expert" if index == 0 else "expert"
                    for index in selected_frames
                ],
                "baseline_mean_iou": float(np.mean(baseline_ious)),
                "best_candidate_frame": best_candidate,
                "best_candidate_prompt_source": "expert",
                "best_candidate_mean_iou": candidate_means[best_candidate],
            }
        )
        if round_number < NUM_ROUNDS:
            selected_frames.append(best_candidate)

    log_payload = {
        "clip": clip_name,
        "ground_truth_source": "expert_annotations",
        "initial_prompt": {"frame": 0, "source": "non_expert_annotations"},
        "correction_prompt_source": "expert_annotations",
        "rounds": selection_log,
        "elapsed_seconds": time.time() - started,
    }
    common.save_json_atomic(output_dir / "logs" / f"{clip_name}.json", log_payload)
    print(f"Completed {clip_name} in {log_payload['elapsed_seconds']:.2f} seconds")


def main() -> None:
    args = parse_args()
    scans_root = args.dataset_root / "micro_ultrasound_scans"
    non_expert_root = args.dataset_root / "non_expert_annotations"
    expert_root = args.dataset_root / "expert_annotations"

    clip_names = list_clip_names(scans_root, non_expert_root, expert_root)
    batches = make_batches(clip_names, args.batch_size)
    if args.list_batches:
        for index, batch in enumerate(batches):
            print(
                f"batch {index}: clips={len(batch)}, first={batch[0]}, last={batch[-1]}"
            )
        return

    if args.batch_index < 0 or args.batch_index >= len(batches):
        raise ValueError(
            f"--batch-index must be between 0 and {len(batches)-1}; "
            f"got {args.batch_index}"
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {args.checkpoint}")

    selected_clips = batches[args.batch_index]
    output_dir = args.output_root / f"mup_mask_prompts_batch_{args.batch_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Total clips: {len(clip_names)}")
    print(f"Batch {args.batch_index}/{len(batches)-1}: {len(selected_clips)} clips")
    print(f"Output: {output_dir}")

    sam2_root = str(args.sam2_root.resolve())
    if sam2_root not in sys.path:
        sys.path.insert(0, sam2_root)
    from sam2.build_sam import build_sam2_video_predictor

    if not common.print_cuda_diagnostics():
        raise RuntimeError("CUDA is unavailable; refusing to run MUP propagation on CPU.")
    device = "cuda"
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
    common.load_reduced_memory_checkpoint(
        predictor, args.checkpoint, args.memory_attention_layers
    )

    batch_started = time.time()
    for position, clip_name in enumerate(selected_clips, start=1):
        print(f"\n[{position}/{len(selected_clips)}] {clip_name}")
        process_clip(
            predictor=predictor,
            clip_name=clip_name,
            scans_root=scans_root,
            non_expert_root=non_expert_root,
            expert_root=expert_root,
            output_dir=output_dir,
            device=device,
            overwrite=args.overwrite,
        )

    print(f"Batch complete in {time.time() - batch_started:.2f} seconds.")


if __name__ == "__main__":
    main()
