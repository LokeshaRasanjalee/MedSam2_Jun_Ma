#!/usr/bin/env python3
"""Create 256x256 VTUS clips with split-specific strides, optionally processing one deterministic video batch."""

from __future__ import annotations

import argparse
import ast
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "datasets/VTUS"
DEFAULT_SPLIT_PATH = (
    DEFAULT_SOURCE_ROOT
    / "VTUS_created_datasets/vtus_len-30_frameinterval-1_interclipstride-4/split_dict_vtus.txt"
)
DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "CMIG_clips/VTUS/vtus_clips_train_stride_15_val_test_stride_30"
)

CLIP_LENGTH = 30
FRAME_INTERVAL = 1
OUTPUT_SIZE = (256, 256)
DEFAULT_BATCH_SIZE = 10
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
SPLIT_STRIDES = {0: 15, 1: 30, 2: 30}
EXPECTED_CLIPS = {0: 285, 1: 30, 2: 75}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--batch-index",
        type=int,
        help="Zero-based batch to process. Omit to process all videos.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--list-batches",
        action="store_true",
        help="Print the deterministic batch assignments and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate sources and count clips without writing output frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output clip folders belonging to this batch if they exist.",
    )
    return parser.parse_args()


def load_split_dict(path: Path) -> dict[int, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"VTUS split file does not exist: {path}")
    raw = ast.literal_eval(path.read_text(encoding="utf-8"))
    split_dict = {int(key): [str(item) for item in value] for key, value in raw.items()}
    if set(split_dict) != {0, 1, 2}:
        raise ValueError(f"Expected split keys 0, 1 and 2; got {sorted(split_dict)}")
    all_ids = [video_id for ids in split_dict.values() for video_id in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("A VTUS video ID occurs in more than one split")
    return split_dict


def split_lookup(split_dict: dict[int, list[str]]) -> dict[str, int]:
    return {
        video_id: split_key
        for split_key, video_ids in split_dict.items()
        for video_id in video_ids
    }


def make_batches(video_ids: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    ordered_ids = sorted(video_ids)
    return [ordered_ids[i : i + batch_size] for i in range(0, len(ordered_ids), batch_size)]


def select_video_ids(
    batches: list[list[str]], batch_index: int | None
) -> list[str]:
    if batch_index is None:
        return [video_id for batch in batches for video_id in batch]
    if batch_index < 0 or batch_index >= len(batches):
        raise ValueError(
            f"--batch-index must be between 0 and {len(batches) - 1}; got {batch_index}"
        )
    return batches[batch_index]


def source_directories(source_root: Path, split_key: int, video_id: str) -> tuple[Path, Path]:
    source_split = "test" if split_key == 2 else "train"
    return (
        source_root / source_split / "images" / video_id,
        source_root / source_split / "masks" / video_id,
    )


def list_masks(mask_dir: Path) -> list[Path]:
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")
    files = sorted(
        path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise RuntimeError(f"No masks found in {mask_dir}")
    return files


def object_flags(mask_files: list[Path]) -> list[bool]:
    flags = []
    for path in mask_files:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {path}")
        flags.append(bool(np.any(mask > 0)))
    return flags


def eligible_sequences(mask_files: list[Path], stride: int) -> list[list[int]]:
    flags = object_flags(mask_files)
    sequences: list[list[int]] = []
    for start_idx in range(0, len(mask_files), stride):
        sequence: list[int] = []
        current_idx = start_idx
        while len(sequence) < CLIP_LENGTH and current_idx < len(mask_files):
            if flags[current_idx]:
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


def prepare_clip_directories(image_dir: Path, mask_dir: Path, overwrite: bool) -> None:
    if (image_dir.exists() or mask_dir.exists()) and not overwrite:
        raise FileExistsError(
            f"Output clip already exists: {image_dir.name}. Use --overwrite to replace it."
        )
    if overwrite:
        shutil.rmtree(image_dir, ignore_errors=True)
        shutil.rmtree(mask_dir, ignore_errors=True)
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)


def write_clip(
    sequence: list[int],
    mask_files: list[Path],
    image_source_dir: Path,
    image_output_dir: Path,
    mask_output_dir: Path,
    overwrite: bool,
) -> None:
    prepare_clip_directories(image_output_dir, mask_output_dir, overwrite)
    try:
        for output_idx, source_idx in enumerate(sequence):
            mask_path = mask_files[source_idx]
            image_path = matching_image(image_source_dir, mask_path)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if image is None or mask is None:
                raise RuntimeError(f"Could not read pair: {image_path}, {mask_path}")
            image = cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST)
            image_destination = image_output_dir / f"{output_idx:04d}.jpg"
            mask_destination = mask_output_dir / f"{output_idx:04d}.png"
            if not cv2.imwrite(str(image_destination), image):
                raise RuntimeError(f"Failed to write image: {image_destination}")
            if not cv2.imwrite(str(mask_destination), mask):
                raise RuntimeError(f"Failed to write mask: {mask_destination}")
    except Exception:
        shutil.rmtree(image_output_dir, ignore_errors=True)
        shutil.rmtree(mask_output_dir, ignore_errors=True)
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

    selected_ids = select_video_ids(batches, args.batch_index)
    batch_label = "all" if args.batch_index is None else str(args.batch_index)
    print(f"Processing batch {batch_label}: {', '.join(selected_ids)}")

    images_output_root = args.dataset_root / "Images"
    masks_output_root = args.dataset_root / "Masks"
    if not args.dry_run:
        images_output_root.mkdir(parents=True, exist_ok=True)
        masks_output_root.mkdir(parents=True, exist_ok=True)
        ensure_output_split(args.split_path, args.dataset_root / "split_dict_vtus.txt")

    counts: Counter[int] = Counter()
    contributing: Counter[int] = Counter()
    for video_id in selected_ids:
        split_key = lookup[video_id]
        stride = SPLIT_STRIDES[split_key]
        image_dir, mask_dir = source_directories(args.source_root, split_key, video_id)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        mask_files = list_masks(mask_dir)
        sequences = eligible_sequences(mask_files, stride)
        if sequences:
            contributing[split_key] += 1

        for sequence in sequences:
            first_frame = mask_files[sequence[0]].stem
            clip_name = f"{video_id}_{CLIP_LENGTH}_{FRAME_INTERVAL}_1_{first_frame}"
            if not args.dry_run:
                write_clip(
                    sequence,
                    mask_files,
                    image_dir,
                    images_output_root / clip_name,
                    masks_output_root / clip_name,
                    args.overwrite,
                )
            counts[split_key] += 1

        print(
            f"[{SPLIT_NAMES[split_key]}] {video_id}: {len(sequences)} clips "
            f"(stride={stride})"
        )

    print(f"\nVTUS batch {batch_label} summary")
    for split_key in (0, 1, 2):
        selected_count = sum(lookup[video_id] == split_key for video_id in selected_ids)
        print(
            f"{SPLIT_NAMES[split_key]}: videos={selected_count}, "
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
