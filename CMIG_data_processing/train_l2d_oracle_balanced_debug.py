#!/usr/bin/env python3
"""Debug variant of L2D training that balances oracle-defer and oracle-stop samples in every training batch."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

import train_l2d_r2plus1d as original


class OracleBalancedBatchSampler(Sampler[list[int]]):
    """Draw a fixed defer/stop composition per batch, with reproducible replacement sampling."""

    def __init__(
        self,
        defer_indices: list[int],
        stop_indices: list[int],
        dataset_size: int,
        batch_size: int,
        defer_fraction: float,
        seed: int,
    ) -> None:
        if not defer_indices or not stop_indices:
            raise RuntimeError(
                "Oracle-balanced sampling requires at least one oracle-defer and one oracle-stop sample"
            )
        self.defer_indices = torch.tensor(defer_indices, dtype=torch.int64)
        self.stop_indices = torch.tensor(stop_indices, dtype=torch.int64)
        self.num_batches = math.ceil(dataset_size / batch_size)
        self.batch_size = batch_size
        self.defer_per_batch = int(round(batch_size * defer_fraction))
        self.defer_per_batch = min(max(self.defer_per_batch, 1), batch_size - 1)
        self.stop_per_batch = batch_size - self.defer_per_batch
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1

        def draw(pool: torch.Tensor, count: int) -> torch.Tensor:
            if len(pool) >= count:
                return pool[torch.randperm(len(pool), generator=generator)[:count]]
            return pool[torch.randint(len(pool), (count,), generator=generator)]

        for _ in range(self.num_batches):
            defer_draw = draw(self.defer_indices, self.defer_per_batch)
            stop_draw = draw(self.stop_indices, self.stop_per_batch)
            batch = torch.cat((defer_draw, stop_draw))
            order = torch.randperm(self.batch_size, generator=generator)
            yield batch[order].tolist()


def oracle_groups(dataset: original.L2DIntermediateDataset, beta: float) -> tuple[list[int], list[int]]:
    defer_indices: list[int] = []
    stop_indices: list[int] = []
    round_counts = {round_number: {"defer": 0, "stop": 0} for round_number in range(1, 5)}
    for index, sample_path in enumerate(dataset.samples):
        with np.load(sample_path, allow_pickle=False) as data:
            action_ious = data["action_ious"].astype(np.float64)
            selected = data["already_prompted_mask"].astype(bool)
            round_number = int(data["round"].item())
        valid = np.concatenate((np.ones(1, dtype=np.bool_), ~selected))
        costs = 1.0 - np.nan_to_num(action_ious, nan=0.0)
        costs[1:] += beta
        costs[~valid] = np.inf
        oracle_defers = int(np.argmin(costs)) > 0
        if oracle_defers:
            defer_indices.append(index)
            round_counts[round_number]["defer"] += 1
        else:
            stop_indices.append(index)
            round_counts[round_number]["stop"] += 1
    print(
        f"Natural training oracle groups: defer={len(defer_indices)}, "
        f"stop={len(stop_indices)}, defer_rate={len(defer_indices) / len(dataset):.4f}"
    )
    for round_number, counts in round_counts.items():
        total = counts["defer"] + counts["stop"]
        print(
            f"  Round {round_number}: defer={counts['defer']}, stop={counts['stop']}, "
            f"defer_rate={counts['defer'] / total:.4f}"
        )
    return defer_indices, stop_indices


def make_balanced_loaders(args: argparse.Namespace) -> dict[str, DataLoader]:
    loaders = {}
    for split in original.SPLITS:
        dataset = original.L2DIntermediateDataset(
            root=args.data_root,
            prompt_dataset=args.prompt_dataset,
            split=split,
            video_key=args.video_key,
            input_channels=args.input_channels,
            architecture=args.architecture,
            horizontal_flip_prob=args.horizontal_flip_prob if split == "train" else 0.0,
        )
        common = {
            "num_workers": args.workers,
            "pin_memory": True,
            "persistent_workers": args.workers > 0,
        }
        if split == "train":
            defer_indices, stop_indices = oracle_groups(dataset, args.beta)
            batch_sampler = OracleBalancedBatchSampler(
                defer_indices=defer_indices,
                stop_indices=stop_indices,
                dataset_size=len(dataset),
                batch_size=args.batch_size,
                defer_fraction=args.train_defer_fraction,
                seed=args.seed,
            )
            print(
                f"Balanced training batches: {batch_sampler.defer_per_batch} oracle-defer + "
                f"{batch_sampler.stop_per_batch} oracle-stop"
            )
            loaders[split] = DataLoader(dataset, batch_sampler=batch_sampler, **common)
        else:
            loaders[split] = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                generator=torch.Generator().manual_seed(args.seed),
                **common,
            )
    return loaders


def extract_debug_args() -> tuple[float, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-defer-fraction", type=float, default=0.5)
    debug_args, remaining = parser.parse_known_args()
    if not 0.0 < debug_args.train_defer_fraction < 1.0:
        raise ValueError("--train-defer-fraction must be strictly between 0 and 1")
    return debug_args.train_defer_fraction, remaining


def main() -> None:
    defer_fraction, remaining = extract_debug_args()
    sys.argv = [sys.argv[0], *remaining]
    original_parse_args = original.parse_args

    def parse_args_with_debug_metadata() -> argparse.Namespace:
        args = original_parse_args()
        args.train_sampling = "oracle_defer_stop_balanced"
        args.train_defer_fraction = defer_fraction
        return args

    original.parse_args = parse_args_with_debug_metadata
    original.make_loaders = make_balanced_loaders
    original.main()


if __name__ == "__main__":
    main()
