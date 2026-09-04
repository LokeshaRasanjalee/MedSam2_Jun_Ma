#!/usr/bin/env python3
"""Build resumable 112x112 VTUS or MUP grayscale L2D samples from merged SAM2 outputs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_SIZE = (112, 112)
SLOTS = 10
ROUNDS = 4
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
SCALE_TAGS = {"10": 1.0, "12": 1.2, "14": 1.4, "18": 1.8}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    clip_root: Path
    propagation_root: Path
    output_root: Path
    image_subdir: str
    split_filename: str
    frame_count: int
    combination_pattern: re.Pattern[str]


CONFIGS = {
    "vtus": DatasetConfig(
        name="vtus",
        clip_root=REPO_ROOT / "CMIG_clips/VTUS/vtus_clips_train_stride_15_val_test_stride_30",
        propagation_root=REPO_ROOT / "CMIG_npz_data/vtus",
        output_root=REPO_ROOT / "CMIG_l2d_data/vtus",
        image_subdir="Images",
        split_filename="split_dict_vtus.txt",
        frame_count=30,
        combination_pattern=re.compile(r"^vtus_(10|12|14|18)_(10|12|14|18)$"),
    ),
    "mup": DatasetConfig(
        name="mup",
        clip_root=REPO_ROOT / "CMIG_clips/MUP/mup_clips_train_stride_5_val_test_stride_10",
        propagation_root=REPO_ROOT / "CMIG_npz_data/mup",
        output_root=REPO_ROOT / "CMIG_l2d_data/mup",
        image_subdir="micro_ultrasound_scans",
        split_filename="split_dict_mup.txt",
        frame_count=10,
        combination_pattern=re.compile(r"^mup_mask_prompts$"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--clip-root", type=Path)
    parser.add_argument("--propagation-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--combination",
        action="append",
        help="VTUS scale tag such as 14_12; repeat as needed. MUP has only mask_prompts.",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="Clips per deterministic batch; 0 means all clips.")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--max-clips", type=int, help="Limit a selected batch for testing.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def effective_config(args: argparse.Namespace) -> DatasetConfig:
    base = CONFIGS[args.dataset]
    return DatasetConfig(
        name=base.name,
        clip_root=(args.clip_root or base.clip_root).resolve(),
        propagation_root=(args.propagation_root or base.propagation_root).resolve(),
        output_root=(args.output_root or base.output_root).resolve(),
        image_subdir=base.image_subdir,
        split_filename=base.split_filename,
        frame_count=base.frame_count,
        combination_pattern=base.combination_pattern,
    )


def load_splits(path: Path) -> tuple[dict[str, str], list[str]]:
    raw = ast.literal_eval(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for split_key, ids in raw.items():
        split = SPLIT_NAMES[int(split_key)]
        for source_id in ids:
            source_id = str(source_id)
            if source_id in mapping:
                raise ValueError(f"Source ID appears in multiple splits: {source_id}")
            mapping[source_id] = split
    return mapping, sorted(mapping, key=len, reverse=True)


def source_id_for_clip(clip: str, ordered_ids: list[str]) -> str:
    matches = [source_id for source_id in ordered_ids if clip.startswith(source_id + "_")]
    if not matches:
        raise ValueError(f"Cannot map clip to a split source ID: {clip}")
    longest = len(matches[0])
    matches = [item for item in matches if len(item) == longest]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous split source ID for {clip}: {matches}")
    return matches[0]


def discover_sources(config: DatasetConfig, requested: list[str] | None) -> list[tuple[str, Path]]:
    discovered: dict[str, Path] = {}
    for path in config.propagation_root.iterdir():
        if not path.is_dir() or config.combination_pattern.fullmatch(path.name) is None:
            continue
        tag = path.name.removeprefix("vtus_") if config.name == "vtus" else "mask_prompts"
        discovered[tag] = path
    if config.name == "mup" and requested and requested != ["mask_prompts"]:
        raise ValueError("MUP supports only --combination mask_prompts")
    selected = sorted(set(requested)) if requested else sorted(discovered)
    missing = sorted(set(selected) - set(discovered))
    if missing:
        raise FileNotFoundError(f"Merged {config.name.upper()} output(s) not found: {missing}")
    if not selected:
        raise FileNotFoundError(f"No merged {config.name.upper()} outputs found in {config.propagation_root}")
    return [(tag, discovered[tag]) for tag in selected]


def select_logs(path: Path, batch_size: int, batch_index: int, max_clips: int | None) -> list[Path]:
    logs = sorted(path.glob("*.json"))
    if batch_size < 0 or batch_index < 0:
        raise ValueError("Batch values cannot be negative")
    if batch_size == 0:
        if batch_index:
            raise ValueError("--batch-index must be 0 when --batch-size is 0")
        selected = logs
    else:
        start = batch_index * batch_size
        selected = logs[start : start + batch_size]
    if max_clips is not None:
        if max_clips < 1:
            raise ValueError("--max-clips must be positive")
        selected = selected[:max_clips]
    return selected


def frame_paths(directory: Path, suffix: str, expected: int) -> list[Path]:
    paths = sorted(directory.glob(f"*{suffix}"))
    if len(paths) != expected:
        raise ValueError(f"Expected {expected} {suffix} files in {directory}; found {len(paths)}")
    return paths


def read_grayscale(directory: Path, locations: np.ndarray, frame_count: int) -> np.ndarray:
    paths = frame_paths(directory, ".jpg", frame_count)
    frames = []
    for location in locations:
        image = cv2.imread(str(paths[int(location)]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Cannot read image: {paths[int(location)]}")
        frames.append(cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_AREA))
    return np.stack(frames)[None].astype(np.uint8, copy=False)


def read_masks(directory: Path, locations: np.ndarray, frame_count: int) -> np.ndarray:
    paths = frame_paths(directory, ".png", frame_count)
    masks = []
    for location in locations:
        path = paths[int(location)]
        # Preserve the label indices in P-mode SAM2 PNGs. OpenCV expands the
        # palette, where both indices are black, and would return an empty mask.
        with Image.open(path) as image:
            labels = np.asarray(image.copy())
        if labels.ndim != 2:
            raise ValueError(f"Expected a 2-D indexed mask in {path}; got {labels.shape}")
        binary = labels > 0
        visible = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
        visible = visible.resize(OUTPUT_SIZE, Image.Resampling.NEAREST)
        masks.append(np.asarray(visible) > 0)
    return np.stack(masks)[None].astype(np.uint8)


def verify_saved_masks(path: Path, expected: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as saved:
        actual = saved["propagated_masks"]
    if actual.dtype != np.uint8 or actual.shape != (1, SLOTS, 112, 112):
        raise RuntimeError(f"Invalid saved propagated_masks in {path}: {actual.dtype} {actual.shape}")
    if not np.array_equal(actual, expected):
        raise RuntimeError(f"Saved propagated_masks differ from decoded source masks: {path}")


def atomic_savez(path: Path, overwrite: bool, **arrays: object) -> str:
    if path.is_file() and not overwrite:
        try:
            with np.load(path, allow_pickle=False) as existing:
                if set(arrays).issubset(existing.files):
                    return "skipped"
        except (OSError, ValueError):
            pass
        raise RuntimeError(f"Existing NPZ is invalid: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def round_targets(path: Path, fixed: list[int], frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    baseline: float | None = None
    candidates: dict[int, float] = {}
    for encoded, values in raw.items():
        key = ast.literal_eval(encoded)
        ious = np.asarray(values, dtype=np.float64)
        if ious.shape != (frame_count,) or not np.isfinite(ious).all():
            raise ValueError(f"Invalid per-frame IoUs for {encoded} in {path}")
        if key == [fixed]:
            baseline = float(ious.mean())
        elif isinstance(key, list) and len(key) == 2 and key[0] == fixed and isinstance(key[1], list) and len(key[1]) == 1:
            candidates[int(key[1][0])] = float(ious.mean())
        else:
            raise ValueError(f"Unexpected action key {encoded} in {path}")
    expected = set(range(SLOTS)) - set(fixed)
    if baseline is None or set(candidates) != expected:
        raise ValueError(f"Incomplete action set in {path}; candidates={sorted(candidates)}, expected={sorted(expected)}")
    action_ious = np.full(1 + SLOTS, np.nan, dtype=np.float32)
    action_ious[0] = baseline
    for slot, value in candidates.items():
        action_ious[slot + 1] = value
    selected = np.zeros(SLOTS, dtype=np.bool_)
    selected[fixed] = True
    return action_ious, selected


def manifest_filename(batch_size: int, batch_index: int, max_clips: int | None) -> str:
    name = f"batch_{batch_index:05d}" if batch_size else "all"
    if max_clips is not None:
        name += f"_first_{max_clips}"
    return name + ".csv"


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["sample_path", "shared_video_path", "clip_name", "source_video_id", "split", "round", "initial_box_scale", "correction_box_scale", "ground_truth_source"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing-{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    config = effective_config(args)
    split_map, ordered_ids = load_splits(config.clip_root / config.split_filename)
    sources = discover_sources(config, args.combination)
    print(f"Dataset: {config.name.upper()}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")
    print(f"Propagation datasets: {len(sources)}")
    reference_clips: list[str] | None = None
    total = 0

    for tag, source_root in sources:
        logs = select_logs(source_root / "logs", args.batch_size, args.batch_index, args.max_clips)
        clip_names = [path.stem for path in logs]
        if reference_clips is None:
            reference_clips = clip_names
        elif clip_names != reference_clips:
            raise RuntimeError(f"Selected clips differ across propagation datasets at {tag}")
        output_name = f"vtus_{tag}" if config.name == "vtus" else "mup_mask_prompts"
        initial_scale = SCALE_TAGS[tag.split("_")[0]] if config.name == "vtus" else np.nan
        correction_scale = SCALE_TAGS[tag.split("_")[1]] if config.name == "vtus" else np.nan
        rows: list[dict[str, object]] = []
        counts = {"written": 0, "skipped": 0}
        samples_with_foreground = 0
        print(f"{output_name}: {len(logs)} clips, {len(logs) * ROUNDS} samples")

        for log_path in logs:
            log = json.loads(log_path.read_text(encoding="utf-8"))
            clip = log_path.stem
            recorded_clip = log.get("video", log.get("clip"))
            if recorded_clip != clip:
                raise ValueError(f"Clip-name mismatch in {log_path}")
            if config.name == "mup":
                if log.get("ground_truth_source") != "expert_annotations":
                    raise ValueError(f"MUP IoUs are not marked as expert-annotation GT in {log_path}")
                locations = np.arange(SLOTS, dtype=np.int16)
                ground_truth_source = "expert_annotations"
            else:
                locations = np.asarray([log["label_to_frame"][str(i)] for i in range(SLOTS)], dtype=np.int16)
                ground_truth_source = "clip_masks"
            if len(set(locations.tolist())) != SLOTS or np.any(locations < 0) or np.any(locations >= config.frame_count):
                raise ValueError(f"Invalid candidate locations in {log_path}: {locations.tolist()}")
            source_id = source_id_for_clip(clip, ordered_ids)
            split = split_map[source_id]
            shared_path = config.output_root / "shared_videos" / f"{clip}.npz"
            if not args.dry_run:
                grayscale = read_grayscale(config.clip_root / config.image_subdir / clip, locations, config.frame_count)
                atomic_savez(
                    shared_path,
                    args.overwrite,
                    grayscale_frames=grayscale,
                    candidate_frame_indices=locations,
                    clip_name=np.asarray(clip),
                    source_frame_count=np.asarray(config.frame_count, dtype=np.int16),
                )

            rounds = log.get("rounds", [])
            if len(rounds) != ROUNDS:
                raise ValueError(f"Expected {ROUNDS} rounds in {log_path}; found {len(rounds)}")
            for round_number, round_log in enumerate(rounds, start=1):
                if int(round_log.get("round", -1)) != round_number:
                    raise ValueError(f"Round ordering mismatch in {log_path}")
                fixed = [int(x) for x in round_log.get("fixed_labels", round_log.get("fixed_frames", []))]
                if len(fixed) != round_number or fixed[0] != 0 or len(set(fixed)) != len(fixed):
                    raise ValueError(f"Invalid selected slots in {log_path}, round {round_number}: {fixed}")
                info_path = source_root / "info_dict" / f"{clip}_round_{round_number}.json"
                action_ious, selected = round_targets(info_path, fixed, config.frame_count)
                if not np.isclose(action_ious[0], float(round_log["baseline_mean_iou"]), atol=1e-7):
                    raise ValueError(f"Baseline IoU mismatch between {log_path} and {info_path}")
                sample_path = config.output_root / output_name / "samples" / f"{clip}_round_{round_number}.npz"
                if not args.dry_run:
                    propagated = read_masks(source_root / "sam2_masks" / f"{clip}_round_{round_number}", locations, config.frame_count)
                    samples_with_foreground += int(propagated.any())
                    result = atomic_savez(
                        sample_path,
                        args.overwrite,
                        propagated_masks=propagated,
                        action_ious=action_ious,
                        already_prompted_mask=selected,
                        candidate_frame_indices=locations,
                        clip_name=np.asarray(clip),
                        source_video_id=np.asarray(source_id),
                        split=np.asarray(split),
                        round=np.asarray(round_number, dtype=np.int8),
                        initial_box_scale=np.asarray(initial_scale, dtype=np.float32),
                        correction_box_scale=np.asarray(correction_scale, dtype=np.float32),
                        ground_truth_source=np.asarray(ground_truth_source),
                    )
                    verify_saved_masks(sample_path, propagated)
                    counts[result] += 1
                rows.append({
                    "sample_path": str(sample_path.relative_to(config.output_root)),
                    "shared_video_path": str(shared_path.relative_to(config.output_root)),
                    "clip_name": clip,
                    "source_video_id": source_id,
                    "split": split,
                    "round": round_number,
                    "initial_box_scale": initial_scale,
                    "correction_box_scale": correction_scale,
                    "ground_truth_source": ground_truth_source,
                })
                total += 1

        if not args.dry_run:
            if rows and samples_with_foreground == 0:
                raise RuntimeError(f"All decoded propagated masks are empty for {output_name}; refusing to accept output")
            manifest = config.output_root / output_name / "manifests" / manifest_filename(args.batch_size, args.batch_index, args.max_clips)
            write_manifest(manifest, rows)
            print(f"  written={counts['written']}, skipped={counts['skipped']}, manifest={manifest}")
            print(f"  samples with foreground={samples_with_foreground}/{len(rows)}")

    verb = "validated" if args.dry_run else "prepared"
    print(f"Successfully {verb} {total} round samples.")


if __name__ == "__main__":
    main()
