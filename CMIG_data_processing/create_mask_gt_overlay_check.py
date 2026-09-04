#!/usr/bin/env python3
"""Create visual overlays comparing indexed SAM2 mask PNGs with frames and GT masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def blend(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    result = image.copy().astype(np.float32)
    result[mask] = (1 - alpha) * result[mask] + alpha * np.asarray(color)
    return result.astype(np.uint8)


def read_labels(path: Path, size: tuple[int, int]) -> np.ndarray:
    # Preserve P-mode label indices. RGB conversion can erase labels when the palette is black.
    image = Image.open(path)
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def main() -> None:
    args = arguments()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.images.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No JPEG frames in {args.images}")

    contact_thumbnails: list[Image.Image] = []
    ious: list[float] = []
    for image_path in image_paths:
        gt_path = args.ground_truth / f"{image_path.stem}.png"
        prediction_path = args.predictions / f"{image_path.stem}.png"
        if not gt_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"Missing GT or prediction for frame {image_path.stem}")
        frame_image = Image.open(image_path).convert("RGB")
        frame = np.asarray(frame_image)
        gt = read_labels(gt_path, frame_image.size)
        prediction = read_labels(prediction_path, frame_image.size)
        intersection = np.logical_and(gt, prediction)
        false_negative = np.logical_and(gt, ~prediction)
        false_positive = np.logical_and(~gt, prediction)
        union = np.logical_or(gt, prediction).sum()
        iou = 1.0 if union == 0 else float(intersection.sum() / union)
        ious.append(iou)

        gt_overlay = blend(frame, gt, (0, 255, 0))
        comparison = blend(frame, intersection, (0, 255, 0))
        comparison = blend(comparison, false_negative, (255, 0, 0))
        comparison = blend(comparison, false_positive, (0, 80, 255))
        panel = Image.new("RGB", (frame.shape[1] * 3, frame.shape[0] + 30), "white")
        panel.paste(Image.fromarray(frame), (0, 30))
        panel.paste(Image.fromarray(gt_overlay), (frame.shape[1], 30))
        panel.paste(Image.fromarray(comparison), (frame.shape[1] * 2, 30))
        draw = ImageDraw.Draw(panel)
        draw.text((5, 8), "Original", fill="black")
        draw.text((frame.shape[1] + 5, 8), "GT (green)", fill="black")
        draw.text((frame.shape[1] * 2 + 5, 8), f"Overlap IoU={iou:.3f}", fill="black")
        destination = output / f"{image_path.stem}_overlay.png"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {destination}")
        panel.save(destination)
        contact_thumbnails.append(panel.resize((576, 214), Image.Resampling.LANCZOS))

    sheet = Image.new("RGB", (576, len(contact_thumbnails) * 214), "white")
    for index, thumbnail in enumerate(contact_thumbnails):
        sheet.paste(thumbnail, (0, index * 214))
    sheet.save(output / "contact_sheet_all_frames.png")
    (output / "legend.txt").write_text(
        "Overlap panel: green=true positive, red=GT missed by SAM2, blue=SAM2 false positive.\n"
        f"Frames={len(ious)}, mean frame IoU={np.mean(ious):.6f}\n"
    )
    print(f"Frames: {len(ious)}")
    print(f"Mean frame IoU: {np.mean(ious):.6f}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
