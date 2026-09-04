#!/usr/bin/env python3
"""Plot training-set no-deferral IoUs against cost-aware oracle-chosen IoUs for one trained configuration."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib_cache"))

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Training run directory containing config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "CMIG_l2d_training/oracle_distribution_plots",
    )
    parser.add_argument("--bins", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run.resolve()
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing training configuration: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    prompt_dataset = str(config["prompt_dataset"])
    alpha = float(config["beta"])
    sample_root = Path(config["data_root"]) / prompt_dataset / "samples"
    if not sample_root.is_dir():
        raise FileNotFoundError(f"Missing intermediate samples: {sample_root}")

    no_deferral_ious: list[float] = []
    oracle_ious: list[float] = []
    oracle_actions: list[int] = []
    rounds: list[int] = []
    for path in sorted(sample_root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if str(data["split"].item()) != "train":
                continue
            action_ious = data["action_ious"].astype(np.float64)
            selected = data["already_prompted_mask"].astype(bool)
            round_number = int(data["round"].item())
        if action_ious.shape != (11,) or selected.shape != (10,):
            raise ValueError(f"Unexpected action shapes in {path}")
        valid = np.concatenate(([True], ~selected))
        costs = 1.0 - np.nan_to_num(action_ious, nan=0.0)
        costs[1:] += alpha
        costs[~valid] = np.inf
        oracle_action = int(np.argmin(costs))
        no_deferral_ious.append(float(action_ious[0]))
        oracle_ious.append(float(action_ious[oracle_action]))
        oracle_actions.append(oracle_action)
        rounds.append(round_number)

    if not no_deferral_ious:
        raise RuntimeError(f"No training samples found in {sample_root}")
    baseline = np.asarray(no_deferral_ious)
    oracle = np.asarray(oracle_ious)
    actions = np.asarray(oracle_actions)
    rounds_array = np.asarray(rounds)
    oracle_deferred = actions > 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_dir.name}_training_oracle_iou_distribution"
    edges = np.linspace(0.0, 1.0, args.bins + 1)
    sample_weight = np.full(baseline.shape, 100.0 / len(baseline))

    fig, axis = plt.subplots(figsize=(9, 5.8), constrained_layout=True)
    axis.hist(
        baseline,
        bins=edges,
        weights=sample_weight,
        histtype="step",
        linewidth=2.5,
        color="#4C78A8",
        label=f"No deferral (mean={baseline.mean():.3f})",
    )
    axis.hist(
        oracle,
        bins=edges,
        weights=sample_weight,
        histtype="step",
        linewidth=2.5,
        color="#E45756",
        label=f"Cost-aware oracle choice (mean={oracle.mean():.3f})",
    )
    axis.axvline(baseline.mean(), color="#4C78A8", linestyle="--", alpha=0.8)
    axis.axvline(oracle.mean(), color="#E45756", linestyle="--", alpha=0.8)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Mean video IoU")
    axis.set_ylabel("Training samples per bin (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.set_title(
        f"Training IoU distribution: {config['dataset']} / {prompt_dataset}\n"
        f"alpha={alpha:g}, oracle deferral rate={100 * oracle_deferred.mean():.1f}%",
        fontweight="bold",
    )
    fig.savefig(args.output_dir / f"{stem}.png", dpi=300)
    fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    csv_path = args.output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_index", "round", "no_deferral_iou", "oracle_action", "oracle_deferred", "oracle_chosen_iou"),
        )
        writer.writeheader()
        for index in range(len(baseline)):
            writer.writerow(
                {
                    "sample_index": index,
                    "round": int(rounds_array[index]),
                    "no_deferral_iou": float(baseline[index]),
                    "oracle_action": int(actions[index]),
                    "oracle_deferred": bool(oracle_deferred[index]),
                    "oracle_chosen_iou": float(oracle[index]),
                }
            )

    summary = {
        "run": run_dir.name,
        "dataset": config["dataset"],
        "prompt_dataset": prompt_dataset,
        "alpha": alpha,
        "training_samples": len(baseline),
        "no_deferral_mean_iou": float(baseline.mean()),
        "oracle_chosen_mean_iou": float(oracle.mean()),
        "mean_iou_gain": float((oracle - baseline).mean()),
        "oracle_deferral_count": int(oracle_deferred.sum()),
        "oracle_non_deferral_count": int((~oracle_deferred).sum()),
        "oracle_deferral_rate": float(oracle_deferred.mean()),
        "per_round": {
            str(round_number): {
                "samples": int((rounds_array == round_number).sum()),
                "no_deferral_mean_iou": float(baseline[rounds_array == round_number].mean()),
                "oracle_chosen_mean_iou": float(oracle[rounds_array == round_number].mean()),
                "oracle_deferral_rate": float(oracle_deferred[rounds_array == round_number].mean()),
            }
            for round_number in range(1, 5)
        },
    }
    with (args.output_dir / f"{stem}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Plot: {args.output_dir / f'{stem}.png'}")


if __name__ == "__main__":
    main()
