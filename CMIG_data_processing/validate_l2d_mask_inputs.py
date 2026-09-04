#!/usr/bin/env python3
"""Fail-fast validation for propagated-mask tensors before L2D training."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    paths = sorted(args.sample_dir.glob("*.npz"))
    if len(paths) != args.expected_count:
        raise RuntimeError(f"Found {len(paths)} samples; expected {args.expected_count}: {args.sample_dir}")
    per_round: Counter[int] = Counter()
    foreground_per_round: Counter[int] = Counter()
    total_foreground = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            masks = data["propagated_masks"]
            round_number = int(data["round"].item())
        if masks.dtype != np.uint8 or masks.shape != (1, 10, 112, 112):
            raise RuntimeError(f"Bad propagated_masks in {path}: {masks.dtype}, {masks.shape}")
        values = np.unique(masks)
        if not set(values.tolist()).issubset({0, 1}):
            raise RuntimeError(f"Non-binary propagated_masks in {path}: {values.tolist()}")
        foreground = int(masks.sum())
        per_round[round_number] += 1
        foreground_per_round[round_number] += foreground
        total_foreground += foreground
    if set(per_round) != {1, 2, 3, 4} or any(foreground_per_round[r] == 0 for r in range(1, 5)):
        raise RuntimeError(f"Missing rounds or an all-empty round: counts={dict(per_round)}, foreground={dict(foreground_per_round)}")
    print(f"Validated {len(paths)} samples in {args.sample_dir}")
    print(f"Samples per round: {dict(sorted(per_round.items()))}")
    print(f"Foreground pixels per round: {dict(sorted(foreground_per_round.items()))}")
    print(f"Total foreground pixels: {total_foreground}")


if __name__ == "__main__":
    main()
