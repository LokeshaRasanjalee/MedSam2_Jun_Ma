#!/usr/bin/env python3
"""
Count number of files inside two expected subfolders for each immediate child folder.

Default expected subfolders: data_pkl, iou_dict

Example:
  python3 count_files_in_two_subfolders.py \
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


def iter_run_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if p.is_dir():
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count files in two subfolders (default: data_pkl and iou_dict) for each folder under ROOT."
    )
    parser.add_argument("root", type=Path, help="Root directory containing multiple run folders.")
    parser.add_argument(
        "--subfolders",
        nargs=2,
        default=["data_pkl", "iou_dict"],
        metavar=("SUBFOLDER_1", "SUBFOLDER_2"),
        help="Names of the two subfolders to count files in (exactly two).",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only count files directly inside each subfolder (do not traverse nested folders).",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write CSV results (columns: folder, subfolder_1, subfolder_2, total).",
    )
    args = parser.parse_args()

    root: Path = args.root
    sub1, sub2 = args.subfolders
    recursive = not args.non_recursive

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"ROOT does not exist or is not a directory: {root}")

    rows = []
    grand1 = grand2 = grandt = 0

    for run_dir in iter_run_dirs(root):
        c1 = count_files(run_dir / sub1, recursive=recursive)
        c2 = count_files(run_dir / sub2, recursive=recursive)
        total = c1 + c2
        rows.append((run_dir.name, c1, c2, total))
        grand1 += c1
        grand2 += c2
        grandt += total

    # Print a simple aligned table.
    name_w = max([len("folder")] + [len(r[0]) for r in rows]) if rows else len("folder")
    header = f"{'folder'.ljust(name_w)}  {sub1:>10}  {sub2:>10}  {'total':>10}"
    print(header)
    print("-" * len(header))
    for name, c1, c2, total in rows:
        print(f"{name.ljust(name_w)}  {c1:10d}  {c2:10d}  {total:10d}")
    print("-" * len(header))
    print(f"{'GRAND_TOTAL'.ljust(name_w)}  {grand1:10d}  {grand2:10d}  {grandt:10d}")

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["folder", sub1, sub2, "total"])
            w.writerows(rows)
            w.writerow(["GRAND_TOTAL", grand1, grand2, grandt])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())








