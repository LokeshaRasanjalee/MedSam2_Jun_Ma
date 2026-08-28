#!/usr/bin/env python3
"""
Count number of files inside `data_npz_4_iou/` for each child folder that ends with `_all`.

Example:
  python3 count_data_npz_4_iou_in_all_folders.py \
    /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/miccai_data_pkl_vtus
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def count_files(path: Path, recursive: bool) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    if recursive:
        return sum(1 for p in path.rglob("*") if p.is_file())
    return sum(1 for p in path.iterdir() if p.is_file())


def iter_all_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.endswith("_all"):
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count files inside data_npz_4_iou for each folder under ROOT that ends with _all."
    )
    parser.add_argument("root", type=Path, help="Root directory containing multiple run folders.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Count files recursively under data_npz_4_iou (default: only direct children).",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write CSV results (columns: folder, data_npz_4_iou).",
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"ROOT does not exist or is not a directory: {root}")

    rows: list[tuple[str, int]] = []
    grand = 0

    for run_dir in iter_all_dirs(root):
        c = count_files(run_dir / "data_npz_4_iou", recursive=args.recursive)
        rows.append((run_dir.name, c))
        grand += c

    name_w = max([len("folder")] + [len(r[0]) for r in rows]) if rows else len("folder")
    header = f"{'folder'.ljust(name_w)}  {'data_npz_4_iou':>14}"
    print(header)
    print("-" * len(header))
    for name, c in rows:
        print(f"{name.ljust(name_w)}  {c:14d}")
    print("-" * len(header))
    print(f"{'GRAND_TOTAL'.ljust(name_w)}  {grand:14d}")

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["folder", "data_npz_4_iou"])
            w.writerows(rows)
            w.writerow(["GRAND_TOTAL", grand])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


