#!/usr/bin/env python3
"""Resumably merge and verify VTUS or MUP propagation batch folders without deleting source batches."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBDIRS = ("info_dict", "sam2_masks", "logs")


@dataclass(frozen=True)
class DatasetConfig:
    root_name: str
    source_pattern: re.Pattern[str]
    combinations: tuple[str, ...]
    batch_count: int
    destination_name: str
    expected_counts: dict[str, int]


SCALE_TAGS = ("10", "12", "14", "18")
CONFIGS = {
    "vtus": DatasetConfig(
        root_name="vtus",
        source_pattern=re.compile(
            r"^vtus_(10|12|14|18)_(10|12|14|18)_batch_(\d+)$"
        ),
        combinations=tuple(
            f"{initial}_{correction}"
            for initial in SCALE_TAGS
            for correction in SCALE_TAGS
        ),
        batch_count=8,
        destination_name="vtus_{combination}",
        expected_counts={
            "info_json": 1560,
            "video_logs": 390,
            "round_mask_dirs": 1560,
            "mask_png": 46800,
        },
    ),
    "mup": DatasetConfig(
        root_name="mup",
        source_pattern=re.compile(r"^mup_mask_prompts_batch_(\d+)$"),
        combinations=("mask_prompts",),
        batch_count=6,
        destination_name="mup_mask_prompts",
        expected_counts={
            "info_json": 1128,
            "video_logs": 282,
            "round_mask_dirs": 1128,
            "mask_png": 11280,
        },
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--combination",
        action="append",
        default=None,
        help="VTUS initial/correction scale tag, such as 14_12.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite-conflicts", action="store_true")
    return parser.parse_args()


def discover_batches(
    root: Path, dataset: str, config: DatasetConfig
) -> dict[str, dict[int, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Propagation root does not exist: {root}")

    discovered: dict[str, dict[int, Path]] = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = config.source_pattern.fullmatch(path.name)
        if match is None:
            continue
        if dataset == "vtus":
            combination = f"{match.group(1)}_{match.group(2)}"
            batch_index = int(match.group(3))
        else:
            combination = "mask_prompts"
            batch_index = int(match.group(1))
        batches = discovered.setdefault(combination, {})
        if batch_index in batches:
            raise RuntimeError(
                f"Duplicate batch {batch_index} for {dataset} {combination}"
            )
        batches[batch_index] = path
    return discovered


def validate_sources(
    discovered: dict[str, dict[int, Path]],
    config: DatasetConfig,
    requested: list[str] | None,
) -> list[str]:
    selected = list(config.combinations) if requested is None else sorted(set(requested))
    invalid = set(selected) - set(config.combinations)
    if invalid:
        raise ValueError(f"Unknown combination(s): {sorted(invalid)}")

    expected_batches = set(range(config.batch_count))
    for combination in selected:
        batches = discovered.get(combination, {})
        if set(batches) != expected_batches:
            raise RuntimeError(
                f"{combination} has batches {sorted(batches)}; "
                f"expected {sorted(expected_batches)}"
            )
        for batch_index, batch_dir in batches.items():
            for subdir in SUBDIRS:
                if not (batch_dir / subdir).is_dir():
                    raise FileNotFoundError(
                        f"Missing {subdir} for {combination} batch {batch_index}: "
                        f"{batch_dir / subdir}"
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
            "Use --overwrite-conflicts only after confirming replacement is intended."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.copying-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copy2(source, temporary)
        if not files_match(source, temporary):
            raise OSError(f"Verification failed after copying {source}")
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
    for subdir in SUBDIRS:
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
            counts[copy_file_atomic(source, destination, overwrite)] += 1
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
            1
            for path in (destination / "sam2_masks").glob("*_round_*")
            if path.is_dir()
        ),
        "mask_png": sum(
            1
            for path in (destination / "sam2_masks").glob("*_round_*/*.png")
            if path.is_file()
        ),
    }


def main() -> None:
    args = parse_args()
    if args.dry_run and args.verify_only:
        raise ValueError("--dry-run and --verify-only cannot be combined")

    config = CONFIGS[args.dataset]
    root = (
        args.root.resolve()
        if args.root is not None
        else REPO_ROOT / "CMIG_npz_data" / config.root_name
    )
    discovered = discover_batches(root, args.dataset, config)
    selected = validate_sources(discovered, config, args.combination)
    mode = "DRY RUN" if args.dry_run else "VERIFY" if args.verify_only else "COPY"
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {mode}")
    print(f"Root: {root}")
    print(f"Combinations: {selected}")

    started = time.time()
    summary: dict[str, object] = {
        "dataset": args.dataset,
        "mode": mode,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "combinations": {},
    }
    for combination in selected:
        destination = root / config.destination_name.format(combination=combination)
        totals = {"source": 0, "copied": 0, "skipped": 0, "missing": 0}
        print(f"\n{combination} -> {destination.name}")
        for batch_index in sorted(discovered[combination]):
            counts = merge_batch(
                discovered[combination][batch_index],
                destination,
                args.dry_run,
                args.verify_only,
                args.overwrite_conflicts,
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
        summary["combinations"][combination] = {
            "file_totals": totals,
            "destination_counts": destination_counts,
        }
        print(f"  totals: {totals}")
        print(f"  destination: {destination_counts}")
        if destination_counts and destination_counts != config.expected_counts:
            raise RuntimeError(
                f"Destination count check failed for {destination}: "
                f"got {destination_counts}, expected {config.expected_counts}"
            )
        if totals["missing"]:
            raise RuntimeError(
                f"Verification found {totals['missing']} missing/mismatched files"
            )

    summary["elapsed_seconds"] = time.time() - started
    if not args.dry_run:
        selection_tag = selected[0] if len(selected) == 1 else "all"
        summary_path = root / (
            f"{args.dataset}_{selection_tag}_merge_{mode.lower()}_summary.json"
        )
        temporary = summary_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        os.replace(temporary, summary_path)
        print(f"\nSummary: {summary_path}")
    print(f"Completed {mode.lower()} in {summary['elapsed_seconds']:.2f} seconds.")


if __name__ == "__main__":
    main()
