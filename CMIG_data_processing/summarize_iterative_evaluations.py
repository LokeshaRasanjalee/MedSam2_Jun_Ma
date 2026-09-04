#!/usr/bin/env python3
"""Merge completed iterative-evaluation clip logs into model and per-iteration CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=REPO_ROOT / "CMIG_iterative_evaluation",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def mean(items: list[float]) -> float:
    return float(np.mean(items)) if items else float("nan")


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".writing-{os.getpid()}")
    fieldnames = list(rows[0])
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.resolve()
    paths = sorted(root.glob(f"*/best_test_chosen_cost/{args.split}/logs/*.json"))
    if not paths:
        raise RuntimeError(f"No iterative clip logs found under {root}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        run_name = path.parents[2].name
        with path.open(encoding="utf-8") as handle:
            item = json.load(handle)
        if len(item.get("mean_iou_trajectory", [])) != 5:
            raise ValueError(f"Expected initial + four iteration IoUs in {path}")
        if len(item.get("correction_count_trajectory", [])) != 5:
            raise ValueError(f"Expected initial + four correction counts in {path}")
        grouped[run_name].append(item)

    model_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    for run_name, items in sorted(grouped.items()):
        config_path = REPO_ROOT / "CMIG_l2d_training" / run_name / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing training config for evaluated run: {config_path}")
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        common_fields = {
            "run_name": run_name,
            "dataset": config["dataset"],
            "prompt_dataset": config["prompt_dataset"],
            "architecture": config.get("architecture", "r2plus1d_18"),
            "loss": config["loss"],
            "seed": config["seed"],
            "beta": config["beta"],
            "checkpoint": "best_test_chosen_cost.pt",
        }
        model_row = {
            **common_fields,
            "clip_count": len(items),
            "initial_mean_iou": mean([item["initial_mean_iou"] for item in items]),
            "final_mean_iou": mean([item["final_mean_iou"] for item in items]),
            "mean_iou_improvement": mean(
                [item["final_mean_iou"] - item["initial_mean_iou"] for item in items]
            ),
            "mean_corrections": mean([item["corrections_used"] for item in items]),
            "rejector_stop_rate": mean(
                [float(item["stop_reason"] == "rejector_stop") for item in items]
            ),
            "mean_accumulated_deferral_cost": mean(
                [item["accumulated_deferral_cost"] for item in items]
            ),
            "mean_final_total_cost": mean([item["final_total_cost"] for item in items]),
            "mean_elapsed_seconds": mean([item["elapsed_seconds"] for item in items]),
        }
        for iteration in range(5):
            model_row[f"iteration_{iteration}_mean_iou"] = mean(
                [item["mean_iou_trajectory"][iteration] for item in items]
            )
            model_row[f"iteration_{iteration}_mean_corrections"] = mean(
                [item["correction_count_trajectory"][iteration] for item in items]
            )
        model_rows.append(model_row)

        for iteration in range(5):
            row = {
                **common_fields,
                "iteration": iteration,
                "clip_count": len(items),
                "mean_iou": mean(
                    [item["mean_iou_trajectory"][iteration] for item in items]
                ),
                "mean_cumulative_corrections": mean(
                    [item["correction_count_trajectory"][iteration] for item in items]
                ),
                "eligible_clip_count": len(items) if iteration == 0 else 0,
                "correction_count": 0,
                "stop_count": 0,
            }
            if iteration > 0:
                decisions = [
                    item["decisions"][iteration - 1]
                    for item in items
                    if len(item["decisions"]) >= iteration
                ]
                row["eligible_clip_count"] = len(decisions)
                row["correction_count"] = sum(
                    decision["action"] == "correct" for decision in decisions
                )
                row["stop_count"] = sum(
                    decision["action"] == "stop" for decision in decisions
                )
            iteration_rows.append(row)

    output_dir = (args.output_dir or root / "summaries").resolve()
    write_csv_atomic(output_dir / f"{args.split}_model_summary.csv", model_rows)
    write_csv_atomic(output_dir / f"{args.split}_iteration_summary.csv", iteration_rows)
    print(f"Models summarized: {len(model_rows)}")
    print(f"Model summary: {output_dir / f'{args.split}_model_summary.csv'}")
    print(f"Iteration summary: {output_dir / f'{args.split}_iteration_summary.csv'}")


if __name__ == "__main__":
    main()
