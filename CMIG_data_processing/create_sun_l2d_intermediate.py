#!/usr/bin/env python3
"""Build resumable 112x112 SUN L2D samples from merged four-round SAM2 outputs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIP_ROOT = REPO_ROOT / "CMIG_clips/SUN/sun_clips_train_stride_30_test_stride_100"
DEFAULT_PROPAGATION_ROOT = REPO_ROOT / "CMIG_npz_data/sunseg"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "CMIG_l2d_data/sunseg"
COMBINATION_PATTERN = re.compile(r"^sunseg_(10|12|14|18)_(10|12|14|18)$")
SCALE_TAGS = {"10": 1.0, "12": 1.2, "14": 1.4, "18": 1.8}
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
OUTPUT_SIZE = (112, 112)
EXPECTED_FRAMES = 100
EXPECTED_SLOTS = 10
EXPECTED_ROUNDS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-root", type=Path, default=DEFAULT_CLIP_ROOT)
    parser.add_argument("--propagation-root", type=Path, default=DEFAULT_PROPAGATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--combination",
        action="append",
        metavar="INITIAL_CORRECTION",
        help="Process one scale tag such as 14_12; repeat to select several. Default: all complete merged folders.",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="Clips per deterministic batch; 0 processes all clips.")
    parser.add_argument("--batch-index", type=int, default=0, help="Zero-based batch index used with --batch-size.")
    parser.add_argument("--max-clips", type=int, help="Limit selected clips for a small validation run.")
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace existing NPZ files.")
    parser.add_argument("--dry-run", action="store_true", help="Validate sources and targets without writing output files.")
    return parser.parse_args()


def load_splits(path: Path) -> tuple[dict[str, str], list[str]]:
    raw = ast.literal_eval(path.read_text(encoding="utf-8"))
    video_to_split: dict[str, str] = {}
    for key, video_ids in raw.items():
        split = SPLIT_NAMES[int(key)]
        for video_id in video_ids:
            if video_id in video_to_split:
                raise ValueError(f"Source video occurs in multiple splits: {video_id}")
            video_to_split[str(video_id)] = split
    return video_to_split, sorted(video_to_split, key=len, reverse=True)


def source_video_for_clip(clip_name: str, ordered_video_ids: list[str]) -> str:
    matches = [video_id for video_id in ordered_video_ids if clip_name.startswith(video_id + "_")]
    if not matches:
        raise ValueError(f"Cannot match clip to SUN split entry: {clip_name}")
    longest = len(matches[0])
    longest_matches = [item for item in matches if len(item) == longest]
    if len(longest_matches) != 1:
        raise ValueError(f"Ambiguous source video for {clip_name}: {longest_matches}")
    return longest_matches[0]


def discover_combinations(root: Path, requested: list[str] | None) -> list[tuple[str, Path]]:
    discovered = {}
    for path in root.iterdir():
        if path.is_dir() and COMBINATION_PATTERN.fullmatch(path.name):
            tag = path.name.removeprefix("sunseg_")
            discovered[tag] = path
    selected = sorted(set(requested)) if requested else sorted(discovered)
    if not selected:
        raise FileNotFoundError(f"No merged SUN scale folders found in {root}")
    missing = [tag for tag in selected if tag not in discovered]
    if missing:
        raise FileNotFoundError(f"Merged SUN combination(s) not found: {missing}")
    return [(tag, discovered[tag]) for tag in selected]


def select_logs(log_dir: Path, batch_size: int, batch_index: int, max_clips: int | None) -> list[Path]:
    logs = sorted(log_dir.glob("*.json"))
    if batch_size < 0 or batch_index < 0:
        raise ValueError("--batch-size and --batch-index cannot be negative")
    if batch_size == 0:
        if batch_index != 0:
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


def numbered_frames(directory: Path, suffix: str) -> list[Path]:
    frames = sorted(directory.glob(f"*{suffix}"))
    if len(frames) != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} {suffix} frames in {directory}, found {len(frames)}")
    return frames


def read_rgb_candidates(directory: Path, locations: np.ndarray) -> np.ndarray:
    paths = numbered_frames(directory, ".jpg")
    output = []
    for location in locations:
        image = cv2.imread(str(paths[int(location)]), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read RGB frame: {paths[int(location)]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
        output.append(np.moveaxis(image, -1, 0))
    return np.stack(output, axis=1).astype(np.uint8, copy=False)


def read_mask_candidates(directory: Path, locations: np.ndarray) -> np.ndarray:
    paths = numbered_frames(directory, ".png")
    output = []
    for location in locations:
        path = paths[int(location)]
        # SAM2 masks are P-mode PNGs whose foreground is palette index 1. Do
        # not use OpenCV here: its palette-to-RGB conversion maps indices 0
        # and 1 to black for these files and silently erases the foreground.
        with Image.open(path) as image:
            labels = np.asarray(image.copy())
        if labels.ndim != 2:
            raise ValueError(f"Expected a 2-D indexed mask in {path}; got {labels.shape}")
        binary = labels > 0
        visible = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
        visible = visible.resize(OUTPUT_SIZE, Image.Resampling.NEAREST)
        output.append(np.asarray(visible) > 0)
    return np.stack(output, axis=0)[None].astype(np.uint8)


def verify_saved_masks(path: Path, expected: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as saved:
        actual = saved["propagated_masks"]
    if actual.dtype != np.uint8 or actual.shape != (1, EXPECTED_SLOTS, 112, 112):
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
        raise RuntimeError(f"Existing NPZ is incomplete or unreadable: {path}; use --overwrite")
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


def parse_round_targets(info_path: Path, fixed_labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    baseline = None
    candidate_means: dict[int, float] = {}
    for encoded_key, frame_ious in info.items():
        key = ast.literal_eval(encoded_key)
        values = np.asarray(frame_ious, dtype=np.float64)
        if values.shape != (EXPECTED_FRAMES,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid IoU vector for {encoded_key} in {info_path}")
        if key == [fixed_labels]:
            baseline = float(values.mean())
        elif isinstance(key, list) and len(key) == 2 and key[0] == fixed_labels and isinstance(key[1], list) and len(key[1]) == 1:
            candidate = int(key[1][0])
            candidate_means[candidate] = float(values.mean())
        else:
            raise ValueError(f"Unexpected key {encoded_key} in {info_path}")
    if baseline is None:
        raise ValueError(f"Missing baseline key for fixed labels {fixed_labels} in {info_path}")
    expected_candidates = set(range(EXPECTED_SLOTS)) - set(fixed_labels)
    if set(candidate_means) != expected_candidates:
        raise ValueError(
            f"Candidate labels in {info_path} are {sorted(candidate_means)}; expected {sorted(expected_candidates)}"
        )
    action_ious = np.full(1 + EXPECTED_SLOTS, np.nan, dtype=np.float32)
    action_ious[0] = baseline
    for candidate, mean_iou in candidate_means.items():
        action_ious[candidate + 1] = mean_iou
    selected = np.zeros(EXPECTED_SLOTS, dtype=np.bool_)
    selected[fixed_labels] = True
    return action_ious, selected


def manifest_name(batch_size: int, batch_index: int, max_clips: int | None) -> str:
    if batch_size:
        name = f"batch_{batch_index:05d}"
    else:
        name = "all"
    if max_clips is not None:
        name += f"_first_{max_clips}"
    return name + ".csv"


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing-{os.getpid()}")
    fields = ["sample_path", "shared_video_path", "clip_name", "source_video_id", "split", "round", "initial_box_scale", "correction_box_scale"]
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
    clip_root = args.clip_root.resolve()
    propagation_root = args.propagation_root.resolve()
    output_root = args.output_root.resolve()
    video_to_split, ordered_video_ids = load_splits(clip_root / "split_dict_sun.txt")
    combinations = discover_combinations(propagation_root, args.combination)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")
    print(f"Combinations: {len(combinations)}")

    reference_names: list[str] | None = None
    total_samples = 0
    for tag, combination_root in combinations:
        initial_tag, correction_tag = tag.split("_")
        logs = select_logs(combination_root / "logs", args.batch_size, args.batch_index, args.max_clips)
        names = [path.stem for path in logs]
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise RuntimeError(f"Selected clip set differs across combinations; first mismatch is {tag}")
        print(f"sunseg_{tag}: {len(logs)} clips, {len(logs) * EXPECTED_ROUNDS} samples")
        rows: list[dict[str, object]] = []
        counts = {"written": 0, "skipped": 0}
        samples_with_foreground = 0
        for log_path in logs:
            log = json.loads(log_path.read_text(encoding="utf-8"))
            clip_name = log_path.stem
            if log.get("video") != clip_name:
                raise ValueError(f"Log video mismatch in {log_path}")
            locations = np.asarray([log["label_to_frame"][str(i)] for i in range(EXPECTED_SLOTS)], dtype=np.int16)
            if locations.shape != (EXPECTED_SLOTS,) or len(set(locations.tolist())) != EXPECTED_SLOTS or np.any(locations < 0) or np.any(locations >= EXPECTED_FRAMES):
                raise ValueError(f"Invalid candidate locations in {log_path}: {locations.tolist()}")
            source_video_id = source_video_for_clip(clip_name, ordered_video_ids)
            split = video_to_split[source_video_id]
            shared_path = output_root / "shared_videos" / f"{clip_name}.npz"
            if not args.dry_run:
                rgb = read_rgb_candidates(clip_root / "Images" / clip_name, locations)
                atomic_savez(
                    shared_path,
                    args.overwrite,
                    rgb_frames=rgb,
                    candidate_frame_indices=locations,
                    clip_name=np.asarray(clip_name),
                    source_frame_count=np.asarray(EXPECTED_FRAMES, dtype=np.int16),
                )
            rounds = log.get("rounds", [])
            if len(rounds) != EXPECTED_ROUNDS:
                raise ValueError(f"Expected four rounds in {log_path}, found {len(rounds)}")
            for round_index, round_log in enumerate(rounds, start=1):
                if int(round_log.get("round", -1)) != round_index:
                    raise ValueError(f"Round order mismatch in {log_path}")
                fixed_labels = [int(value) for value in round_log["fixed_labels"]]
                if len(fixed_labels) != round_index or fixed_labels[0] != 0 or len(set(fixed_labels)) != len(fixed_labels):
                    raise ValueError(f"Invalid fixed labels for round {round_index} in {log_path}: {fixed_labels}")
                info_path = combination_root / "info_dict" / f"{clip_name}_round_{round_index}.json"
                action_ious, selected = parse_round_targets(info_path, fixed_labels)
                if not np.isclose(action_ious[0], float(round_log["baseline_mean_iou"]), atol=1e-7):
                    raise ValueError(f"Baseline mean IoU mismatch between {info_path} and {log_path}")
                sample_path = output_root / f"sunseg_{tag}" / "samples" / f"{clip_name}_round_{round_index}.npz"
                if not args.dry_run:
                    masks = read_mask_candidates(
                        combination_root / "sam2_masks" / f"{clip_name}_round_{round_index}", locations
                    )
                    samples_with_foreground += int(masks.any())
                    result = atomic_savez(
                        sample_path,
                        args.overwrite,
                        propagated_masks=masks,
                        action_ious=action_ious,
                        already_prompted_mask=selected,
                        candidate_frame_indices=locations,
                        clip_name=np.asarray(clip_name),
                        source_video_id=np.asarray(source_video_id),
                        split=np.asarray(split),
                        round=np.asarray(round_index, dtype=np.int8),
                        initial_box_scale=np.asarray(SCALE_TAGS[initial_tag], dtype=np.float32),
                        correction_box_scale=np.asarray(SCALE_TAGS[correction_tag], dtype=np.float32),
                    )
                    verify_saved_masks(sample_path, masks)
                    counts[result] += 1
                rows.append(
                    {
                        "sample_path": str(sample_path.relative_to(output_root)),
                        "shared_video_path": str(shared_path.relative_to(output_root)),
                        "clip_name": clip_name,
                        "source_video_id": source_video_id,
                        "split": split,
                        "round": round_index,
                        "initial_box_scale": SCALE_TAGS[initial_tag],
                        "correction_box_scale": SCALE_TAGS[correction_tag],
                    }
                )
                total_samples += 1
        if not args.dry_run:
            if rows and samples_with_foreground == 0:
                raise RuntimeError(f"All decoded propagated masks are empty for sunseg_{tag}; refusing to accept output")
            manifest = output_root / f"sunseg_{tag}" / "manifests" / manifest_name(args.batch_size, args.batch_index, args.max_clips)
            write_manifest(manifest, rows)
            print(f"  written={counts['written']}, skipped={counts['skipped']}, manifest={manifest}")
            print(f"  samples with foreground={samples_with_foreground}/{len(rows)}")

    print(f"Successfully validated {total_samples} round samples." if args.dry_run else f"Successfully prepared {total_samples} round samples.")


if __name__ == "__main__":
    main()
