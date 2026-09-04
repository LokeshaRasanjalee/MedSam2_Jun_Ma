#!/usr/bin/env python3
"""Train L2D with 50% oracle-defer samples and round-balanced oracle-stop samples."""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

import train_l2d_r2plus1d as original


class RoundAwareBatchSampler(Sampler[list[int]]):
    """Build 8-sample batches: four defers plus one stop from each round."""

    def __init__(
        self,
        groups: dict[tuple[int, str], list[int]],
        dataset_size: int,
        seed: int,
        round2_defer_probability: float,
    ) -> None:
        required = [(r, "stop") for r in range(1, 5)] + [(1, "defer"), (2, "defer")]
        missing = [group for group in required if not groups.get(group)]
        if missing:
            raise RuntimeError(f"Round-aware sampling requires non-empty groups; missing {missing}")
        self.groups = {key: torch.tensor(value, dtype=torch.int64) for key, value in groups.items()}
        self.num_batches = math.ceil(dataset_size / 8)
        self.seed = seed
        self.round2_defer_probability = round2_defer_probability
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    @staticmethod
    def draw(pool: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
        if count == 0:
            return torch.empty(0, dtype=torch.int64)
        if len(pool) >= count:
            return pool[torch.randperm(len(pool), generator=generator)[:count]]
        return pool[torch.randint(len(pool), (count,), generator=generator)]

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_batches):
            # Four defer draws. Each is independently drawn from round 2 with
            # probability p and otherwise from round 1 (90/10 by default).
            round2_count = int(
                (torch.rand(4, generator=generator) < self.round2_defer_probability).sum()
            )
            round1_count = 4 - round2_count
            parts = [
                self.draw(self.groups[(1, "defer")], round1_count, generator),
                self.draw(self.groups[(2, "defer")], round2_count, generator),
            ]
            # The stop half is exactly round-balanced: one stop from each round.
            for round_number in range(1, 5):
                parts.append(self.draw(self.groups[(round_number, "stop")], 1, generator))
            batch = torch.cat(parts)
            yield batch[torch.randperm(8, generator=generator)].tolist()


def oracle_groups(dataset: original.L2DIntermediateDataset, beta: float) -> dict[tuple[int, str], list[int]]:
    groups = {(r, decision): [] for r in range(1, 5) for decision in ("defer", "stop")}
    for index, sample_path in enumerate(dataset.samples):
        with np.load(sample_path, allow_pickle=False) as data:
            action_ious = data["action_ious"].astype(np.float64)
            selected = data["already_prompted_mask"].astype(bool)
            round_number = int(data["round"].item())
        valid = np.concatenate((np.ones(1, dtype=np.bool_), ~selected))
        costs = 1.0 - np.nan_to_num(action_ious, nan=0.0)
        costs[1:] += beta
        costs[~valid] = np.inf
        decision = "defer" if int(np.argmin(costs)) > 0 else "stop"
        groups[(round_number, decision)].append(index)
    print("Natural training oracle groups by round:")
    for round_number in range(1, 5):
        defer_count = len(groups[(round_number, "defer")])
        stop_count = len(groups[(round_number, "stop")])
        total = defer_count + stop_count
        print(
            f"  Round {round_number}: defer={defer_count}, stop={stop_count}, "
            f"defer_rate={defer_count / total:.4f}"
        )
    return groups


def make_round_aware_loaders(args: argparse.Namespace) -> dict[str, DataLoader]:
    if args.batch_size != 8:
        raise ValueError("Round-aware sampling currently requires --batch-size 8")
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
            sampler = RoundAwareBatchSampler(
                groups=oracle_groups(dataset, args.beta),
                dataset_size=len(dataset),
                seed=args.seed,
                round2_defer_probability=args.round2_defer_probability,
            )
            loaders[split] = DataLoader(dataset, batch_sampler=sampler, **common)
            print(
                "Round-aware batches: 4 defers (round-1/round-2 probabilities "
                f"{1 - args.round2_defer_probability:.2f}/{args.round2_defer_probability:.2f}) + "
                "one stop from each round"
            )
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


def extract_sampling_args() -> tuple[float, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--round2-defer-probability", type=float, default=0.10)
    sampling_args, remaining = parser.parse_known_args()
    if not 0.0 <= sampling_args.round2_defer_probability <= 1.0:
        raise ValueError("--round2-defer-probability must be in [0,1]")
    return sampling_args.round2_defer_probability, remaining


def main() -> None:
    probability, remaining = extract_sampling_args()
    sys.argv = [sys.argv[0], *remaining]
    original_parse_args = original.parse_args

    def parse_args_with_sampling_metadata() -> argparse.Namespace:
        args = original_parse_args()
        args.train_sampling = "round_aware_oracle_balanced"
        args.round2_defer_probability = probability
        args.train_defer_fraction = 0.5
        args.stop_samples_per_round = 1
        return args

    original.parse_args = parse_args_with_sampling_metadata
    original.make_loaders = make_round_aware_loaders
    original.main()


if __name__ == "__main__":
    main()
