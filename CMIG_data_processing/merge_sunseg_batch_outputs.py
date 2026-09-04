#!/usr/bin/env python3
"""Merge each SUN prompt-scale combination's 15 batch folders into one resumable, verified output folder without deleting the sources."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "CMIG_npz_data/sunseg"
SOURCE_PATTERN = re.compile(
    r"^sunseg_(10|12|14|18)_(10|12|14|18)_batch_(\d+)$"
)
EXPECTED_BATCHES = set(range(15))
EXPECTED_SUBDIRS = ("info_dict", "sam2_masks", "logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--combination",
        action="append",
        default=None,
        metavar="INITIAL_CORRECTION",
        help="Merge only one combination, such as 14_12. Repeat for several.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show the planned merges without copying files.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify already merged destinations without copying files.",
    )
    parser.add_argument(
        "--overwrite-conflicts",
        action="store_true",
        help="Replace destination files whose size or modification time differs.",
    )
    return parser.parse_args()


def discover_batches(root: Path) -> dict[str, dict[int, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"SUN propagation root does not exist: {root}")

    combinations: dict[str, dict[int, Path]] = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = SOURCE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        combination = f"{match.group(1)}_{match.group(2)}"
        batch_index = int(match.group(3))
        if batch_index in combinations.setdefault(combination, {}):
            raise RuntimeError(
                f"Duplicate batch {batch_index} for combination {combination}"
            )
        combinations[combination][batch_index] = path
    return combinations


def validate_sources(
    combinations: dict[str, dict[int, Path]], requested: list[str] | None
) -> list[str]:
    expected_combinations = {
        f"{initial}_{correction}"
        for correction in (10, 12, 14, 18)
        for initial in (10, 12, 14, 18)
    }
    if requested is None:
        selected = sorted(expected_combinations)
    else:
        selected = sorted(set(requested))
        invalid = set(selected) - expected_combinations
        if invalid:
            raise ValueError(f"Unknown prompt-scale combination(s): {sorted(invalid)}")

    for combination in selected:
        batches = combinations.get(combination, {})
        batch_indices = set(batches)
        if batch_indices != EXPECTED_BATCHES:
            raise RuntimeError(
                f"Combination {combination} has batches {sorted(batch_indices)}; "
                f"expected {sorted(EXPECTED_BATCHES)}"
            )
        for batch_index, batch_dir in batches.items():
            for subdir in EXPECTED_SUBDIRS:
                path = batch_dir / subdir
                if not path.is_dir():
                    raise FileNotFoundError(
                        f"Missing {subdir} for {combination} batch {batch_index}: {path}"
                    )
    return selected


def files_match(source: Path, destination: Path) -> bool:
    if not destination.is_file():
        return False
    source_stat = source.stat()
    destination_stat = destination.stat()
    return (
        source_stat.st_size == destination_stat.st_size
        and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
    )


def copy_file_atomic(source: Path, destination: Path, overwrite: bool) -> str:
    if files_match(source, destination):
        return "skipped"
    if destination.exists() and not overwrite:
        raise RuntimeError(
            f"Conflicting destination file: {destination}\n"
            "Rerun with --overwrite-conflicts only if replacing it is intentional."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.copying-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copy2(source, temporary)
        if not files_match(source, temporary):
            raise OSError(f"Verification failed after copying {source} to {temporary}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "copied"


def merge_batch(
    source_batch: Path,
    destination_root: Path,
    dry_run: bool,
    verify_only: bool,
    overwrite: bool,
) -> dict[str, int]:
    counts = {"source": 0, "copied": 0, "skipped": 0, "missing": 0}
    for subdir in EXPECTED_SUBDIRS:
        source_root = source_batch / subdir
        for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
            counts["source"] += 1
            destination = destination_root / subdir / source.relative_to(source_root)
            if dry_run:
                continue
            if verify_only:
                if files_match(source, destination):
                    counts["skipped"] += 1
                else:
                    counts["missing"] += 1
                continue
            result = copy_file_atomic(source, destination, overwrite)
            counts[result] += 1
    return counts


def count_destination(destination: Path) -> dict[str, int]:
    return {
        "info_json": sum(
            1 for path in (destination / "info_dict").glob("*.json") if path.is_file()
        ),
        "video_logs": sum(
            1 for path in (destination / "logs").glob("*.json") if path.is_file()
        ),
        "round_mask_dirs": sum(
            1 for path in (destination / "sam2_masks").glob("*_round_*") if path.is_dir()
        ),
        "mask_png": sum(
            1 for path in (destination / "sam2_masks").glob("*_round_*/*.png") if path.is_file()
        ),
    }


def main() -> None:
    args = parse_args()
    if args.dry_run and args.verify_only:
        raise ValueError("--dry-run and --verify-only cannot be used together")

    root = args.root.resolve()
    discovered = discover_batches(root)
    selected = validate_sources(discovered, args.combination)
    mode = "DRY RUN" if args.dry_run else "VERIFY" if args.verify_only else "COPY"
    print(f"Mode: {mode}")
    print(f"Root: {root}")
    print(f"Combinations: {len(selected)}")

    run_started = time.time()
    run_summary: dict[str, object] = {
        "mode": mode,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "combinations": {},
    }

    for combination in selected:
        destination = root / f"sunseg_{combination}"
        totals = {"source": 0, "copied": 0, "skipped": 0, "missing": 0}
        print(f"\n{combination} -> {destination.name}")
        for batch_index in sorted(discovered[combination]):
            counts = merge_batch(
                discovered[combination][batch_index],
                destination,
                dry_run=args.dry_run,
                verify_only=args.verify_only,
                overwrite=args.overwrite_conflicts,
            )
            for key, value in counts.items():
                totals[key] += value
            print(
                f"  batch {batch_index:02d}: files={counts['source']}, "
                f"copied={counts['copied']}, skipped={counts['skipped']}, "
                f"missing_or_mismatch={counts['missing']}",
                flush=True,
            )

        destination_counts = {} if args.dry_run else count_destination(destination)
        run_summary["combinations"][combination] = {
            "file_totals": totals,
            "destination_counts": destination_counts,
        }
        print(f"  totals: {totals}")
        if destination_counts:
            print(f"  destination: {destination_counts}")
            expected = {
                "info_json": 1500,
                "video_logs": 375,
                "round_mask_dirs": 1500,
                "mask_png": 150000,
            }
            if destination_counts != expected:
                raise RuntimeError(
                    f"Destination count check failed for {destination}: "
                    f"got {destination_counts}, expected {expected}"
                )
            if totals["missing"]:
                raise RuntimeError(
                    f"Verification found {totals['missing']} missing or mismatched files "
                    f"for {combination}"
                )

    run_summary["elapsed_seconds"] = time.time() - run_started
    if not args.dry_run:
        selection_tag = selected[0] if len(selected) == 1 else "all"
        summary_path = root / (
            f"sunseg_{selection_tag}_merge_{mode.lower()}_summary.json"
        )
        temporary = summary_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(run_summary, handle, indent=2)
        os.replace(temporary, summary_path)
        print(f"\nSummary: {summary_path}")
    print(f"Completed {mode.lower()} in {run_summary['elapsed_seconds']:.2f} seconds.")


if __name__ == "__main__":
    main()
