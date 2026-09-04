#!/usr/bin/env python3
"""Iteratively evaluate a trained L2D rejector by alternating real SAM2 propagation and rejector-selected correction prompts."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSING_ROOT = Path(__file__).resolve().parent
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

import propagate_mup_mask_prompts_4rounds as mup_workflow
import propagate_sun_box_prompts_4rounds as common
import train_l2d_r2plus1d as training


# SAM2 is identical for every rejector run. These are intentionally not model-run arguments.
SAM2_ROOT = REPO_ROOT / "sam2"
SAM2_CHECKPOINT = SAM2_ROOT / "checkpoints/sam2.1_hiera_tiny.pt"
SAM2_CONFIG_PATH = SAM2_ROOT / "sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_HYDRA_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_IMAGE_SIZE = 256
SAM2_MEMORY_ATTENTION_LAYERS = 2

DATASET_LAYOUTS = {
    "sunseg": {
        "clip_root": REPO_ROOT / "CMIG_clips/SUN/sun_clips_train_stride_30_test_stride_100",
        "frames": "Images",
        "gt": "Masks",
        "prompt_type": "box",
    },
    "vtus": {
        "clip_root": REPO_ROOT / "CMIG_clips/VTUS/vtus_clips_train_stride_15_val_test_stride_30",
        "frames": "Images",
        "gt": "Masks",
        "prompt_type": "box",
    },
    "mup": {
        "clip_root": REPO_ROOT / "CMIG_clips/MUP/mup_clips_train_stride_5_val_test_stride_10",
        "frames": "micro_ultrasound_scans",
        "gt": "expert_annotations",
        "initial_prompts": "non_expert_annotations",
        "prompt_type": "mask",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Training run directory or a checkpoint .pt file. A run directory selects best_val_chosen_cost.pt.",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--max-corrections", type=int, default=4)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--dataset-root", type=Path, help="Optional override for the generated clip dataset root.")
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "CMIG_iterative_evaluation"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-save-masks", action="store_true")
    parser.add_argument("--list-batches", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(model_path: Path) -> tuple[Path, Path]:
    model_path = model_path.resolve()
    if model_path.is_dir():
        run_dir = model_path
        checkpoint = run_dir / "checkpoints/best_val_chosen_cost.pt"
    else:
        checkpoint = model_path
        if checkpoint.parent.name != "checkpoints":
            raise ValueError("A checkpoint file must be inside its training run's checkpoints/ directory")
        run_dir = checkpoint.parent.parent
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Rejector checkpoint does not exist: {checkpoint}")
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {config_path}")
    return checkpoint, run_dir


def load_training_configuration(checkpoint_path: Path, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with (run_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    if isinstance(checkpoint_args, Namespace):
        checkpoint_args = vars(checkpoint_args)
    if not isinstance(checkpoint_args, dict):
        raise TypeError("Checkpoint 'args' must be a dictionary or argparse Namespace")

    # Old R(2+1)D runs predate the explicit architecture field.
    config.setdefault("architecture", checkpoint_args.get("architecture", "r2plus1d_18"))
    for key in ("dataset", "prompt_dataset", "input_channels", "video_key"):
        checkpoint_value = checkpoint_args.get(key)
        config_value = config.get(key)
        if checkpoint_value is not None and config_value is not None and str(checkpoint_value) != str(config_value):
            raise RuntimeError(
                f"Training configuration mismatch for {key}: config.json={config_value!r}, "
                f"checkpoint={checkpoint_value!r}"
            )
        if config_value is None and checkpoint_value is not None:
            config[key] = checkpoint_value
    required = ("dataset", "prompt_dataset", "architecture", "input_channels", "video_key", "beta")
    missing = [key for key in required if key not in config]
    if missing:
        raise RuntimeError(f"Training configuration is missing required fields: {missing}")
    return config, checkpoint


def prompt_scales(dataset: str, prompt_dataset: str) -> tuple[float | None, float | None]:
    if dataset == "mup":
        if prompt_dataset != "mup_mask_prompts":
            raise ValueError(f"Unexpected MUP prompt dataset: {prompt_dataset}")
        return None, None
    prefix = f"{dataset}_"
    if not prompt_dataset.startswith(prefix):
        raise ValueError(f"Cannot infer prompt scales from {prompt_dataset!r}")
    tags = prompt_dataset[len(prefix) :].split("_")
    if len(tags) != 2 or not all(tag.isdigit() for tag in tags):
        raise ValueError(f"Prompt dataset must end in initial/correction scale tags: {prompt_dataset}")
    return int(tags[0]) / 10.0, int(tags[1]) / 10.0


def test_clip_names(data_root: Path, prompt_dataset: str, split: str) -> list[str]:
    sample_root = data_root / prompt_dataset / "samples"
    if not sample_root.is_dir():
        raise FileNotFoundError(f"Intermediate sample directory does not exist: {sample_root}")
    clips: set[str] = set()
    # Every clip has four independent round samples; round 1 alone is enough
    # to recover the unique split membership without opening the same clip four times.
    for path in sorted(sample_root.glob("*_round_1.npz")):
        with np.load(path, allow_pickle=False) as data:
            if str(data["split"].item()) == split:
                clips.add(str(data["clip_name"].item()))
    if not clips:
        raise RuntimeError(f"No {split} clips found under {sample_root}")
    return sorted(clips)


def make_batches(names: list[str], batch_size: int) -> list[list[str]]:
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    return [names[index : index + batch_size] for index in range(0, len(names), batch_size)]


def build_rejector(config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_args = Namespace(
        architecture=str(config["architecture"]),
        input_channels=int(config["input_channels"]),
        gru_hidden_size=int(config.get("gru_hidden_size", 256)),
        gru_unidirectional=bool(config.get("gru_unidirectional", False)),
        pretrained_weights=None,
        no_pretrained=True,
    )
    model = training.build_model(model_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model


def normalize_frames(frames: np.ndarray, architecture: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(frames, copy=True)).float().div_(255.0)
    if tensor.shape[0] == 3:
        if architecture == "resnet18_gru":
            mean_values, std_values = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean_values, std_values = (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)
        mean = torch.tensor(mean_values).view(3, 1, 1, 1)
        std = torch.tensor(std_values).view(3, 1, 1, 1)
        return (tensor - mean) / std
    if tensor.shape[0] == 1:
        mean, std = (0.449, 0.226) if architecture == "resnet18_gru" else (0.400, 0.225)
        return (tensor - mean) / std
    raise ValueError(f"Unexpected shared-video channel count: {tensor.shape[0]}")


def rejector_input(
    normalized_frames: torch.Tensor,
    predictions: dict[int, np.ndarray],
    candidate_locations: np.ndarray,
) -> torch.Tensor:
    masks = []
    for frame_index in candidate_locations.tolist():
        if int(frame_index) not in predictions:
            raise RuntimeError(f"SAM2 did not return candidate frame {frame_index}")
        mask = predictions[int(frame_index)].astype(np.uint8)
        masks.append(cv2.resize(mask, (112, 112), interpolation=cv2.INTER_NEAREST))
    mask_tensor = torch.from_numpy(np.stack(masks, axis=0)[None]).float()
    return torch.cat((normalized_frames, mask_tensor), dim=0).unsqueeze(0)


def choose_action(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    selected_slots: list[int],
    device: torch.device,
) -> tuple[int, list[float], list[float | None]]:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        frame_scores = model(model_input.to(device, non_blocking=True))[0]
    raw = frame_scores.detach().float().cpu().tolist()
    full_scores = torch.cat((torch.zeros(1, device=device), frame_scores))
    for slot in selected_slots:
        full_scores[slot + 1] = torch.inf
    action = int(full_scores.argmin().item())
    masked: list[float | None] = [0.0]
    masked.extend(None if index in selected_slots else float(score) for index, score in enumerate(raw))
    return action, [float(value) for value in raw], masked


def save_state_masks(
    output_dir: Path,
    clip: str,
    state_number: int,
    predictions: dict[int, np.ndarray],
    frame_stems: list[str],
    palette: list[int] | None,
) -> None:
    common.save_masks_atomic(
        output_dir / "sam2_masks" / clip / f"state_{state_number}",
        predictions,
        frame_stems,
        palette,
    )


def process_clip(
    clip: str,
    config: dict[str, Any],
    layout: dict[str, Any],
    clip_root: Path,
    data_root: Path,
    predictor: Any,
    rejector: torch.nn.Module,
    device: torch.device,
    initial_scale: float | None,
    correction_scale: float | None,
    max_corrections: int,
    beta: float,
    output_dir: Path,
    save_masks: bool,
    overwrite: bool,
) -> dict[str, Any]:
    log_path = output_dir / "logs" / f"{clip}.json"
    if log_path.is_file() and not overwrite:
        with log_path.open(encoding="utf-8") as handle:
            print(f"Skipping complete clip: {clip}")
            return json.load(handle)

    frame_dir = clip_root / str(layout["frames"]) / clip
    gt_dir = clip_root / str(layout["gt"]) / clip
    frame_paths = sorted(path for path in frame_dir.iterdir() if path.suffix.lower() == ".jpg")
    if not frame_paths:
        raise RuntimeError(f"No JPEG frames found for {clip}")
    frame_stems = [path.stem for path in frame_paths]
    gt_masks, palette = common.load_binary_masks(gt_dir, frame_stems)

    shared_path = data_root / "shared_videos" / f"{clip}.npz"
    with np.load(shared_path, allow_pickle=False) as shared:
        frames = shared[str(config["video_key"])].copy()
        locations = shared["candidate_frame_indices"].astype(np.int64)
    if locations.shape != (10,) or len(set(locations.tolist())) != 10:
        raise ValueError(f"Invalid candidate locations in {shared_path}: {locations}")
    normalized_frames = normalize_frames(frames, str(config["architecture"]))

    prompt_type = str(layout["prompt_type"])
    selected_slots = [0]
    if prompt_type == "box":
        assert initial_scale is not None and correction_scale is not None
        slot_to_frame = {slot: int(frame) for slot, frame in enumerate(locations)}
        boxes = {
            slot: common.scaled_box(
                gt_masks[frame], initial_scale if slot == 0 else correction_scale
            )
            for slot, frame in slot_to_frame.items()
        }
        propagate = lambda: common.propagate_with_boxes(
            predictor, inference_state, selected_slots, slot_to_frame, boxes, "cuda"
        )
        prompt_sources = {slot: "gt_box" for slot in range(10)}
    else:
        initial_dir = clip_root / str(layout["initial_prompts"]) / clip
        non_expert_masks, _ = common.load_binary_masks(initial_dir, frame_stems)
        prompt_masks = {int(locations[0]): non_expert_masks[int(locations[0])]}
        prompt_masks.update({int(locations[slot]): gt_masks[int(locations[slot])] for slot in range(1, 10)})
        propagate = lambda: mup_workflow.propagate_with_masks(
            predictor,
            inference_state,
            [int(locations[slot]) for slot in selected_slots],
            prompt_masks,
            "cuda",
        )
        prompt_sources = {0: "non_expert_mask", **{slot: "expert_mask" for slot in range(1, 10)}}

    started = time.time()
    with common.autocast_context("cuda"):
        inference_state = predictor.init_state(video_path=str(frame_dir))
    predictions = propagate()
    ious = common.per_frame_ious(predictions, gt_masks)
    initial_mean_iou = float(np.mean(ious))
    if save_masks:
        save_state_masks(output_dir, clip, 0, predictions, frame_stems, palette)

    decisions: list[dict[str, Any]] = []
    mean_iou_trajectory = [initial_mean_iou]
    correction_count_trajectory = [0]
    stop_reason = "maximum_corrections"
    for iteration in range(1, max_corrections + 1):
        iteration_started = time.time()
        input_tensor = rejector_input(normalized_frames, predictions, locations)
        action, raw_scores, masked_scores = choose_action(
            rejector, input_tensor, selected_slots, device
        )
        before_iou = float(np.mean(ious))
        decision: dict[str, Any] = {
            "iteration": iteration,
            "selected_slots_before_decision": selected_slots.copy(),
            "selected_frames_before_decision": [int(locations[slot]) for slot in selected_slots],
            "raw_frame_scores": raw_scores,
            "valid_action_scores": masked_scores,
            "mean_iou_before_action": before_iou,
        }
        if action == 0:
            decision.update(
                {
                    "action": "stop",
                    "chosen_action_index": 0,
                    "realized_chosen_cost": 1.0 - before_iou,
                    "iteration_seconds": time.time() - iteration_started,
                }
            )
            decisions.append(decision)
            stop_reason = "rejector_stop"
            break

        selected_slot = action - 1
        selected_slots.append(selected_slot)
        predictions = propagate()
        ious = common.per_frame_ious(predictions, gt_masks)
        after_iou = float(np.mean(ious))
        mean_iou_trajectory.append(after_iou)
        correction_count_trajectory.append(len(selected_slots) - 1)
        if save_masks:
            save_state_masks(
                output_dir, clip, len(selected_slots) - 1, predictions, frame_stems, palette
            )
        decision.update(
            {
                "action": "correct",
                "chosen_action_index": action,
                "selected_candidate_slot": selected_slot,
                "selected_frame": int(locations[selected_slot]),
                "prompt_source": prompt_sources[selected_slot],
                "mean_iou_after_action": after_iou,
                "iou_change": after_iou - before_iou,
                "realized_chosen_cost": 1.0 - after_iou + beta,
                "iteration_seconds": time.time() - iteration_started,
            }
        )
        decisions.append(decision)

    corrections = len(selected_slots) - 1
    final_mean_iou = float(np.mean(ious))
    # A stop decision ends interaction for the clip. Carry its unchanged final
    # state forward so cohort-level results remain comparable at iterations 1-4.
    while len(mean_iou_trajectory) < max_corrections + 1:
        mean_iou_trajectory.append(final_mean_iou)
        correction_count_trajectory.append(corrections)
    result = {
        "clip": clip,
        "dataset": config["dataset"],
        "candidate_frame_indices": locations.tolist(),
        "prompt_type": prompt_type,
        "initial_prompt": {
            "candidate_slot": 0,
            "frame": int(locations[0]),
            "source": prompt_sources[0],
            "box_scale": initial_scale,
        },
        "correction_box_scale": correction_scale,
        "beta": beta,
        "max_corrections": max_corrections,
        "corrections_used": corrections,
        "initial_mean_iou": initial_mean_iou,
        "final_mean_iou": final_mean_iou,
        "mean_iou_trajectory": mean_iou_trajectory,
        "correction_count_trajectory": correction_count_trajectory,
        "accumulated_deferral_cost": corrections * beta,
        "final_total_cost": 1.0 - final_mean_iou + corrections * beta,
        "stop_reason": stop_reason,
        "selected_candidate_slots": selected_slots,
        "selected_frames": [int(locations[slot]) for slot in selected_slots],
        "per_frame_final_ious": ious,
        "decisions": decisions,
        "elapsed_seconds": time.time() - started,
    }
    common.save_json_atomic(log_path, result)
    return result


def main() -> None:
    args = parse_args()
    if args.max_corrections < 1 or args.max_corrections > 9:
        raise ValueError("--max-corrections must be between 1 and 9")
    checkpoint_path, run_dir = resolve_checkpoint(args.model)
    config, checkpoint = load_training_configuration(checkpoint_path, run_dir)
    dataset = str(config["dataset"])
    if dataset not in DATASET_LAYOUTS:
        raise ValueError(f"Unsupported stored dataset: {dataset}")
    defaults = training.DATASET_DEFAULTS[dataset]
    data_root = Path(config.get("data_root", defaults["root"])).resolve()
    clip_root = (args.dataset_root or DATASET_LAYOUTS[dataset]["clip_root"]).resolve()
    initial_scale, correction_scale = prompt_scales(dataset, str(config["prompt_dataset"]))

    all_clips = test_clip_names(data_root, str(config["prompt_dataset"]), args.split)
    batches = make_batches(all_clips, args.batch_size)
    if args.list_batches:
        for index, batch in enumerate(batches):
            print(f"batch {index}: clips={len(batch)}, first={batch[0]}, last={batch[-1]}")
        return
    if args.batch_index < 0 or args.batch_index >= len(batches):
        raise ValueError(f"--batch-index must be in [0, {len(batches)-1}]")
    selected_clips = batches[args.batch_index]

    if not SAM2_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Common SAM2 checkpoint does not exist: {SAM2_CHECKPOINT}")
    if not SAM2_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Common SAM2 config does not exist: {SAM2_CONFIG_PATH}")
    if not clip_root.is_dir():
        raise FileNotFoundError(f"Clip dataset root does not exist: {clip_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for iterative SAM2/rejector evaluation")
    device = torch.device("cuda:0")

    checkpoint_tag = checkpoint_path.stem
    output_dir = args.output_root.resolve() / run_dir.name / checkpoint_tag / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training run: {run_dir}")
    print(f"Rejector checkpoint: {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"Dataset/prompt data: {dataset}/{config['prompt_dataset']}")
    print(f"Architecture: {config['architecture']}, channels={config['input_channels']}")
    print(f"Prompt scales: initial={initial_scale}, correction={correction_scale}")
    print(f"Beta: {config['beta']}")
    print(f"Common SAM2 checkpoint: {SAM2_CHECKPOINT}")
    print(f"Common SAM2 config: {SAM2_CONFIG_PATH}")
    print(f"SAM2 image size/layers: {SAM2_IMAGE_SIZE}/{SAM2_MEMORY_ATTENTION_LAYERS}")
    print(f"Split clips: {len(all_clips)}; batch {args.batch_index}/{len(batches)-1}: {len(selected_clips)}")
    print(f"Output: {output_dir}")

    rejector = build_rejector(config, checkpoint, device)
    sam2_root_string = str(SAM2_ROOT.resolve())
    if sam2_root_string not in sys.path:
        sys.path.insert(0, sam2_root_string)
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        config_file=SAM2_HYDRA_CONFIG,
        ckpt_path=None,
        device="cuda",
        hydra_overrides_extra=[
            f"++model.image_size={SAM2_IMAGE_SIZE}",
            f"++model.memory_attention.num_layers={SAM2_MEMORY_ATTENTION_LAYERS}",
        ],
    )
    common.load_reduced_memory_checkpoint(
        predictor, SAM2_CHECKPOINT, SAM2_MEMORY_ATTENTION_LAYERS
    )

    results = []
    for position, clip in enumerate(selected_clips, start=1):
        print(f"[{position}/{len(selected_clips)}] {clip}", flush=True)
        result = process_clip(
            clip=clip,
            config=config,
            layout=DATASET_LAYOUTS[dataset],
            clip_root=clip_root,
            data_root=data_root,
            predictor=predictor,
            rejector=rejector,
            device=device,
            initial_scale=initial_scale,
            correction_scale=correction_scale,
            max_corrections=args.max_corrections,
            beta=float(config["beta"]),
            output_dir=output_dir,
            save_masks=not args.no_save_masks,
            overwrite=args.overwrite,
        )
        results.append(result)

    iteration_summary = {}
    for iteration in range(0, args.max_corrections + 1):
        iteration_summary[str(iteration)] = {
            "mean_iou": float(
                np.mean([item["mean_iou_trajectory"][iteration] for item in results])
            ),
            "mean_cumulative_corrections": float(
                np.mean(
                    [item["correction_count_trajectory"][iteration] for item in results]
                )
            ),
        }
        if iteration > 0:
            eligible = [
                item["decisions"][iteration - 1]
                for item in results
                if len(item["decisions"]) >= iteration
            ]
            iteration_summary[str(iteration)].update(
                {
                    "eligible_clip_count": len(eligible),
                    "correction_count": sum(
                        decision["action"] == "correct" for decision in eligible
                    ),
                    "stop_count": sum(
                        decision["action"] == "stop" for decision in eligible
                    ),
                }
            )
    summary = {
        "batch_index": args.batch_index,
        "clip_count": len(results),
        "mean_initial_iou": float(np.mean([item["initial_mean_iou"] for item in results])),
        "mean_final_iou": float(np.mean([item["final_mean_iou"] for item in results])),
        "mean_corrections": float(np.mean([item["corrections_used"] for item in results])),
        "mean_accumulated_deferral_cost": float(
            np.mean([item["accumulated_deferral_cost"] for item in results])
        ),
        "mean_final_total_cost": float(np.mean([item["final_total_cost"] for item in results])),
        "rejector_stop_rate": float(
            np.mean([item["stop_reason"] == "rejector_stop" for item in results])
        ),
        "iterations": iteration_summary,
    }
    common.save_json_atomic(output_dir / "batch_summaries" / f"batch_{args.batch_index}.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
