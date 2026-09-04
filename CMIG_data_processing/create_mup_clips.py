#!/usr/bin/env python3
"""Create 256x256 MUP clips from paired expert/non-expert masks, with optional deterministic batch processing."""

from __future__ import annotations

import argparse
import ast
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "datasets/Micro_Ultrasound_Prostate_Segmentation_Dataset/MUP_created_datasets"
)
DEFAULT_SPLIT_PATH = REPO_ROOT / "miccai_data_pkl_mup/mask_k10_mup_all/split_dict.txt"
DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "CMIG_clips/MUP/mup_clips_train_stride_5_val_test_stride_10"
)

CLIP_LENGTH = 10
FRAME_INTERVAL = 1
OUTPUT_SIZE = (256, 256)
DEFAULT_BATCH_SIZE = 10
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
SPLIT_STRIDES = {0: 5, 1: 10, 2: 10}
EXPECTED_CLIPS = {0: 248, 1: 25, 2: 25}
VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--batch-index",
        type=int,
        help="Zero-based batch to process. Omit to process all subjects.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--list-batches",
        action="store_true",
        help="Print deterministic subject assignments and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and count clips without writing frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output clip folders belonging to the selected batch.",
    )
    return parser.parse_args()


def load_split_dict(path: Path) -> dict[int, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"MUP split file does not exist: {path}")
    raw = ast.literal_eval(path.read_text(encoding="utf-8"))
    split_dict = {int(key): [str(item).zfill(2) for item in values] for key, values in raw.items()}
    if set(split_dict) != {0, 1, 2}:
        raise ValueError(f"Expected split keys 0, 1 and 2; got {sorted(split_dict)}")
    all_ids = [subject for values in split_dict.values() for subject in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("A MUP subject occurs in more than one split")
    return split_dict


def split_lookup(split_dict: dict[int, list[str]]) -> dict[str, int]:
    return {
        subject: split_key
        for split_key, subjects in split_dict.items()
        for subject in subjects
    }


def make_batches(subjects: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    ordered = sorted(subjects)
    return [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]


def select_subjects(batches: list[list[str]], batch_index: int | None) -> list[str]:
    if batch_index is None:
        return [subject for batch in batches for subject in batch]
    if batch_index < 0 or batch_index >= len(batches):
        raise ValueError(
            f"--batch-index must be between 0 and {len(batches) - 1}; got {batch_index}"
        )
    return batches[batch_index]


def source_directories(source_root: Path, subject: str) -> tuple[Path, Path, Path]:
    return (
        source_root / "micro_ultrasound_scans" / f"microUS_train_{subject}",
        source_root / "expert_annotations" / f"expert_annotation_train_{subject}",
        source_root / "non_expert_annotations" / f"nonexpert_annotation_train_{subject}",
    )


def indexed_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {directory}")
    files = {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    }
    if not files:
        raise RuntimeError(f"No supported files found in {directory}")
    return files


def common_mask_files(expert_dir: Path, nonexpert_dir: Path) -> list[tuple[Path, Path]]:
    expert = indexed_files(expert_dir)
    nonexpert = indexed_files(nonexpert_dir)
    common_names = sorted(expert.keys() & nonexpert.keys())
    if not common_names:
        raise RuntimeError(f"No matching expert/non-expert masks in {expert_dir} and {nonexpert_dir}")
    return [(expert[name], nonexpert[name]) for name in common_names]


def eligible_sequences(mask_pairs: list[tuple[Path, Path]], stride: int) -> list[list[int]]:
    valid_flags: list[bool] = []
    for expert_path, nonexpert_path in mask_pairs:
        expert = cv2.imread(str(expert_path), cv2.IMREAD_GRAYSCALE)
        nonexpert = cv2.imread(str(nonexpert_path), cv2.IMREAD_GRAYSCALE)
        if expert is None or nonexpert is None:
            raise RuntimeError(f"Could not read mask pair: {expert_path}, {nonexpert_path}")
        valid_flags.append(bool(np.any(expert > 0) and np.any(nonexpert > 0)))

    sequences: list[list[int]] = []
    for start_idx in range(0, len(mask_pairs), stride):
        sequence: list[int] = []
        current_idx = start_idx
        while len(sequence) < CLIP_LENGTH and current_idx < len(mask_pairs):
            if valid_flags[current_idx]:
                sequence.append(current_idx)
            current_idx += FRAME_INTERVAL
        if len(sequence) == CLIP_LENGTH:
            sequences.append(sequence)
    return sequences


def matching_image(image_dir: Path, mask_path: Path) -> Path:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = image_dir / f"{mask_path.stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No source image matches mask {mask_path}")


def prepare_clip_directories(output_dirs: tuple[Path, Path, Path], overwrite: bool) -> None:
    if any(path.exists() for path in output_dirs) and not overwrite:
        raise FileExistsError(
            f"Output clip already exists: {output_dirs[0].name}. Use --overwrite to replace it."
        )
    if overwrite:
        for path in output_dirs:
            shutil.rmtree(path, ignore_errors=True)
    for path in output_dirs:
        path.mkdir(parents=True)


def write_clip(
    sequence: list[int],
    mask_pairs: list[tuple[Path, Path]],
    image_source_dir: Path,
    output_dirs: tuple[Path, Path, Path],
    overwrite: bool,
) -> None:
    prepare_clip_directories(output_dirs, overwrite)
    image_output_dir, expert_output_dir, nonexpert_output_dir = output_dirs
    try:
        for output_idx, source_idx in enumerate(sequence):
            expert_path, nonexpert_path = mask_pairs[source_idx]
            image_path = matching_image(image_source_dir, expert_path)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            expert = cv2.imread(str(expert_path), cv2.IMREAD_UNCHANGED)
            nonexpert = cv2.imread(str(nonexpert_path), cv2.IMREAD_UNCHANGED)
            if image is None or expert is None or nonexpert is None:
                raise RuntimeError(
                    f"Could not read image/masks: {image_path}, {expert_path}, {nonexpert_path}"
                )
            image = cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            expert = cv2.resize(expert, OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST)
            nonexpert = cv2.resize(nonexpert, OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST)
            stem = f"{output_idx:04d}"
            if not cv2.imwrite(str(image_output_dir / f"{stem}.jpg"), image):
                raise RuntimeError(f"Failed to write image for {image_path}")
            if not cv2.imwrite(str(expert_output_dir / f"{stem}.png"), expert):
                raise RuntimeError(f"Failed to write expert mask for {expert_path}")
            if not cv2.imwrite(str(nonexpert_output_dir / f"{stem}.png"), nonexpert):
                raise RuntimeError(f"Failed to write non-expert mask for {nonexpert_path}")
    except Exception:
        for path in output_dirs:
            shutil.rmtree(path, ignore_errors=True)
        raise


def ensure_output_split(source_path: Path, destination: Path) -> None:
    source_text = source_path.read_text(encoding="utf-8")
    if destination.exists():
        if destination.read_text(encoding="utf-8") != source_text:
            raise RuntimeError(f"Output split differs from source split: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(source_text)
    except FileExistsError:
        if destination.read_text(encoding="utf-8") != source_text:
            raise RuntimeError(f"Output split differs from source split: {destination}")


def main() -> None:
    args = parse_args()
    split_dict = load_split_dict(args.split_path)
    lookup = split_lookup(split_dict)
    batches = make_batches(list(lookup), args.batch_size)

    if args.list_batches:
        for index, batch in enumerate(batches):
            print(f"batch {index}: {', '.join(batch)}")
        return

    selected = select_subjects(batches, args.batch_index)
    batch_label = "all" if args.batch_index is None else str(args.batch_index)
    print(f"Processing batch {batch_label}: {', '.join(selected)}")

    images_output_root = args.dataset_root / "micro_ultrasound_scans"
    expert_output_root = args.dataset_root / "expert_annotations"
    nonexpert_output_root = args.dataset_root / "non_expert_annotations"
    if not args.dry_run:
        for path in (images_output_root, expert_output_root, nonexpert_output_root):
            path.mkdir(parents=True, exist_ok=True)
        ensure_output_split(args.split_path, args.dataset_root / "split_dict_mup.txt")

    counts: Counter[int] = Counter()
    contributing: Counter[int] = Counter()
    for subject in selected:
        split_key = lookup[subject]
        stride = SPLIT_STRIDES[split_key]
        image_dir, expert_dir, nonexpert_dir = source_directories(args.source_root, subject)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        mask_pairs = common_mask_files(expert_dir, nonexpert_dir)
        sequences = eligible_sequences(mask_pairs, stride)
        if sequences:
            contributing[split_key] += 1

        for sequence in sequences:
            first_frame = mask_pairs[sequence[0]][0].stem
            clip_name = f"{subject}_{CLIP_LENGTH}_{FRAME_INTERVAL}_{stride}_{first_frame}"
            if not args.dry_run:
                write_clip(
                    sequence,
                    mask_pairs,
                    image_dir,
                    (
                        images_output_root / clip_name,
                        expert_output_root / clip_name,
                        nonexpert_output_root / clip_name,
                    ),
                    args.overwrite,
                )
            counts[split_key] += 1
        print(
            f"[{SPLIT_NAMES[split_key]}] {subject}: {len(sequences)} clips "
            f"(stride={stride})"
        )

    print(f"\nMUP batch {batch_label} summary")
    for split_key in (0, 1, 2):
        video_count = sum(lookup[subject] == split_key for subject in selected)
        print(
            f"{SPLIT_NAMES[split_key]}: videos={video_count}, "
            f"contributing_videos={contributing[split_key]}, clips={counts[split_key]}"
        )
    print(f"Total clips in this batch: {sum(counts.values())}")

    if args.batch_index is None:
        for split_key, expected in EXPECTED_CLIPS.items():
            if counts[split_key] != expected:
                raise RuntimeError(
                    f"Unexpected {SPLIT_NAMES[split_key]} count: "
                    f"{counts[split_key]} instead of {expected}"
                )
        print(f"Full-dataset counts match the expected total of {sum(EXPECTED_CLIPS.values())}.")


if __name__ == "__main__":
    main()
