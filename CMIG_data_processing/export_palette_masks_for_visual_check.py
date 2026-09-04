#!/usr/bin/env python3
"""Export indexed segmentation masks as ordinary black/white PNGs for visual verification."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder containing indexed PNG masks.")
    parser.add_argument("output", type=Path, help="Separate output folder; source files are never changed.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be separate from, and not inside, the source folder")
    paths = sorted(source.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG masks found in {source}")
    output.mkdir(parents=True, exist_ok=True)

    thumbnails: list[tuple[str, Image.Image]] = []
    nonempty = 0
    for path in paths:
        # np.asarray on a PIL P-mode image preserves label indices; converting to RGB first would not.
        labels = np.asarray(Image.open(path))
        visible = ((labels > 0).astype(np.uint8) * 255)
        nonempty += int(np.any(visible))
        destination = output / path.name
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {destination}")
        image = Image.fromarray(visible, mode="L")
        image.save(destination)
        thumb = image.resize((128, 128), Image.Resampling.NEAREST).convert("RGB")
        thumbnails.append((path.stem, thumb))

    columns = 10
    cell_width, cell_height = 128, 148
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, thumbnail) in enumerate(thumbnails):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(thumbnail, (x, y))
        draw.text((x + 4, y + 130), label, fill="black")
    sheet.save(output / "contact_sheet.png")

    print(f"Source masks: {len(paths)}")
    print(f"Non-empty masks: {nonempty}")
    print(f"Visible masks: {output}")
    print(f"Contact sheet: {output / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
