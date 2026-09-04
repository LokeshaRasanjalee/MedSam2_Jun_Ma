#!/usr/bin/env python3
"""Create the CMIG SUN clips using split-specific strides and resize every image and mask to 256x256."""

from __future__ import annotations

import argparse
import ast
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "CMIG_clips/SUN/sun_clips_train_stride_30_test_stride_100"
)
DEFAULT_IMAGE_ROOTS = [
    REPO_ROOT / "datasets/SUN_data/home/lokesha/Downloads/sundatabase_positive_part1",
    REPO_ROOT / "datasets/SUN_data/home/lokesha/Downloads/sundatabase_positive_part2",
]
DEFAULT_TRAIN_MASK_ROOT = (
    REPO_ROOT
    / "datasets/SUN_data/SUN-SEG-Annotation/SUN-SEG-Annotation/TrainDataset/GT"
)
DEFAULT_TEST_MASK_ROOT = (
    REPO_ROOT
    / "datasets/SUN_data/SUN-SEG-Annotation/SUN-SEG-Annotation/TestHardDataset/Unseen/GT"
)

CLIP_LENGTH = 100
FRAME_INTERVAL = 1
OUTPUT_SIZE = (256, 256)
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
SPLIT_STRIDES = {0: 30, 1: 100, 2: 100}
EXPECTED_CLIPS = {0: 278, 1: 24, 2: 73}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-roots",
        type=Path,
        nargs="+",
        default=DEFAULT_IMAGE_ROOTS,
        help="One or more SUN RGB roots; both positive_part1 and positive_part2 are used by default.",
    )
    parser.add_argument("--train-mask-root", type=Path, default=DEFAULT_TRAIN_MASK_ROOT)
    parser.add_argument("--test-mask-root", type=Path, default=DEFAULT_TEST_MASK_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace clip folders that already exist in Images/ and Masks/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count clips and validate source files without writing output frames.",
    )
    return parser.parse_args()


def load_split_dict(path: Path) -> dict[int, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = ast.literal_eval(handle.read())

    split_dict = {int(key): [str(item) for item in value] for key, value in raw.items()}
    if set(split_dict) != {0, 1, 2}:
        raise ValueError(f"Expected split keys 0, 1 and 2 in {path}; got {sorted(split_dict)}")

    all_ids = [item for values in split_dict.values() for item in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError(f"A SUN video ID occurs in more than one split in {path}")
    return split_dict


def validate_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def has_object(mask_path: Path) -> bool:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")
    return bool(np.any(mask > 0))


def source_image_directory(image_roots: list[Path], video_id: str) -> Path:
    # The original SUN images are grouped by base case (e.g. case20_17 -> case20).
    base_case = video_id.split("_", 1)[0]
    matches = [root / base_case for root in image_roots if (root / base_case).is_dir()]
    if not matches:
        searched = ", ".join(str(root / base_case) for root in image_roots)
        raise FileNotFoundError(f"Image directory for {video_id} was not found; searched: {searched}")
    if len(matches) > 1:
        raise RuntimeError(f"Image case {base_case} occurs in multiple RGB roots: {matches}")
    return matches[0]


def source_image_name(mask_name: str) -> str:
    return str(Path(mask_name).with_suffix(".jpg"))


def frame_number(filename: str) -> str:
    final_component = Path(filename).stem.split("_")[-1]
    if final_component.startswith("image"):
        final_component = final_component[len("image") :]
    return final_component


def eligible_sequences(mask_files: list[Path], stride: int) -> list[list[int]]:
    sequences: list[list[int]] = []
    object_flags = [has_object(path) for path in mask_files]

    for start_idx in range(0, len(mask_files), stride):
        sequence: list[int] = []
        current_idx = start_idx
        while len(sequence) < CLIP_LENGTH and current_idx < len(mask_files):
            if object_flags[current_idx]:
                sequence.append(current_idx)
            current_idx += FRAME_INTERVAL
        if len(sequence) == CLIP_LENGTH:
            sequences.append(sequence)
    return sequences


def read_and_resize_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read source image: {path}")
    return cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)


def read_and_resize_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read source mask: {path}")
    return cv2.resize(mask, OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST)


def prepare_clip_directories(
    image_dir: Path, mask_dir: Path, overwrite: bool
) -> None:
    existing = image_dir.exists() or mask_dir.exists()
    if existing and not overwrite:
        raise FileExistsError(
            f"Output clip already exists: {image_dir.name}. "
            "Use --overwrite to replace it."
        )
    if overwrite:
        if image_dir.exists():
            shutil.rmtree(image_dir)
        if mask_dir.exists():
            shutil.rmtree(mask_dir)
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
            image_path = image_source_dir / source_image_name(mask_path.name)
            image = read_and_resize_image(image_path)
            mask = read_and_resize_mask(mask_path)

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


def main() -> None:
    args = parse_args()
    split_path = args.dataset_root / "split_dict_sun.txt"

    for image_root in args.image_roots:
        validate_directory(image_root, "SUN image root")
    validate_directory(args.train_mask_root, "SUN train/validation mask root")
    validate_directory(args.test_mask_root, "SUN test mask root")
    if not split_path.is_file():
        raise FileNotFoundError(f"SUN split file does not exist: {split_path}")

    split_dict = load_split_dict(split_path)
    images_output_root = args.dataset_root / "Images"
    masks_output_root = args.dataset_root / "Masks"
    if not args.dry_run:
        images_output_root.mkdir(parents=True, exist_ok=True)
        masks_output_root.mkdir(parents=True, exist_ok=True)

    clip_counts: Counter[int] = Counter()
    contributing_videos: Counter[int] = Counter()

    for split_key in (0, 1, 2):
        split_name = SPLIT_NAMES[split_key]
        stride = SPLIT_STRIDES[split_key]
        mask_root = args.test_mask_root if split_key == 2 else args.train_mask_root

        for video_id in split_dict[split_key]:
            video_mask_dir = mask_root / video_id
            image_source_dir = source_image_directory(args.image_roots, video_id)
            validate_directory(video_mask_dir, f"Mask directory for {video_id}")
            validate_directory(image_source_dir, f"Image directory for {video_id}")

            mask_files = sorted(
                path
                for path in video_mask_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            sequences = eligible_sequences(mask_files, stride)
            if sequences:
                contributing_videos[split_key] += 1

            for clip_idx, sequence in enumerate(sequences, start=1):
                first_frame = frame_number(mask_files[sequence[0]].name)
                clip_name = f"{video_id}_{clip_idx}_{first_frame}"
                if not args.dry_run:
                    write_clip(
                        sequence=sequence,
                        mask_files=mask_files,
                        image_source_dir=image_source_dir,
                        image_output_dir=images_output_root / clip_name,
                        mask_output_dir=masks_output_root / clip_name,
                        overwrite=args.overwrite,
                    )
                clip_counts[split_key] += 1

            print(
                f"[{split_name}] {video_id}: {len(sequences)} clips "
                f"(stride={stride})"
            )

    print("\nSUN CMIG clip summary")
    for split_key in (0, 1, 2):
        split_name = SPLIT_NAMES[split_key]
        actual = clip_counts[split_key]
        expected = EXPECTED_CLIPS[split_key]
        print(
            f"{split_name}: videos={len(split_dict[split_key])}, "
            f"contributing_videos={contributing_videos[split_key]}, "
            f"clips={actual}, expected={expected}"
        )
        if actual != expected:
            raise RuntimeError(
                f"Unexpected {split_name} clip count: generated {actual}, expected {expected}"
            )

    mode = "validated" if args.dry_run else "created"
    print(f"Successfully {mode} {sum(clip_counts.values())} clips.")


if __name__ == "__main__":
    main()
