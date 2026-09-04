#!/usr/bin/env python3
"""Plot Figure 3 prompt-scale and Figure 4 alpha analyses from completed iterative test evaluations."""

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


EVALUATION_ROOT = REPO_ROOT / "CMIG_iterative_evaluation"
OUTPUT_ROOT = EVALUATION_ROOT / "figures"
SCALES = (1.0, 1.2, 1.4, 1.8)
ALPHAS = tuple(index / 10 for index in range(8))
EXPECTED_CLIPS = {"sunseg": 73, "vtus": 75, "mup": 25}
COLORS = {"mup": "#1f77b4", "sunseg": "#ff7f0e", "vtus": "#2ca02c"}
LABELS = {"mup": "MUP", "sunseg": "SUN-SEG", "vtus": "VTUS"}


def scale_tag(value: float) -> str:
    return str(int(round(value * 10)))


def alpha_tag(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def figure3_run(dataset: str, initial: float, correction: float) -> str:
    prompt = f"{dataset}_{scale_tag(initial)}_{scale_tag(correction)}"
    if initial == 1.4 and correction == 1.0:
        return f"{prompt}_alpha_010_loss_mae_seed_42"
    return f"{prompt}_arch_r2plus1d_18_alpha_010_loss_mae_seed_42"


def figure4_run(dataset: str, alpha: float) -> str:
    prompt = {
        "sunseg": "sunseg_14_10",
        "vtus": "vtus_14_10",
        "mup": "mup_mask_prompts",
    }[dataset]
    tag = alpha_tag(alpha)
    if tag == "010":
        return f"{prompt}_alpha_010_loss_mae_seed_42"
    return f"{prompt}_arch_r2plus1d_18_alpha_{tag}_loss_mae_seed_42"


def load_run(run_name: str, dataset: str) -> list[dict]:
    log_root = EVALUATION_ROOT / run_name / "best_test_chosen_cost/test/logs"
    paths = sorted(log_root.glob("*.json"))
    expected = EXPECTED_CLIPS[dataset]
    if len(paths) != expected:
        raise RuntimeError(f"{run_name}: expected {expected} clip logs, found {len(paths)}")
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if len(record.get("mean_iou_trajectory", [])) != 5:
            raise RuntimeError(f"{path}: expected initial plus four iteration IoUs")
        records.append(record)
    return records


def annotate_heatmap(axis, values: np.ndarray, image, decimals: int) -> None:
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    midpoint = (low + high) / 2
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if value < midpoint else "black"
            axis.text(
                column,
                row,
                f"{value:.{decimals}f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )


def plot_figure3(iteration: int) -> None:
    rows = []
    matrices = {}
    for dataset in ("sunseg", "vtus"):
        final_iou = np.zeros((len(SCALES), len(SCALES)), dtype=float)
        relative_gain = np.zeros_like(final_iou)
        # Rows are displayed from correction scale 1.8 down to 1.0, matching the paper.
        correction_order = tuple(reversed(SCALES))
        for row, correction in enumerate(correction_order):
            for column, initial in enumerate(SCALES):
                run_name = figure3_run(dataset, initial, correction)
                records = load_run(run_name, dataset)
                initial_mean = float(np.mean([item["mean_iou_trajectory"][0] for item in records]))
                final_mean = float(
                    np.mean([item["mean_iou_trajectory"][iteration] for item in records])
                )
                gain = 100.0 * (final_mean - initial_mean) / initial_mean if initial_mean else 0.0
                final_iou[row, column] = final_mean
                relative_gain[row, column] = gain
                rows.append(
                    {
                        "dataset": dataset,
                        "initial_prompt_scale": initial,
                        "correction_prompt_scale": correction,
                        "initial_mean_iou": initial_mean,
                        "evaluation_iteration": iteration,
                        "mean_iou_after_selected_iteration": final_mean,
                        "relative_iou_gain_percent": gain,
                        "run_name": run_name,
                    }
                )
        matrices[dataset] = (final_iou, relative_gain, correction_order)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), constrained_layout=True)
    for row, dataset in enumerate(("sunseg", "vtus")):
        final_iou, relative_gain, correction_order = matrices[dataset]
        for column, (values, title, cmap, decimals) in enumerate(
            (
                (final_iou, "Final Mean IoU", "viridis", 3),
                (relative_gain, "Relative IoU Gain (%)", "magma", 1),
            )
        ):
            axis = axes[row, column]
            image = axis.imshow(values, cmap=cmap, aspect="equal")
            annotate_heatmap(axis, values, image, decimals)
            axis.set_xticks(range(len(SCALES)), [f"{value:.1f}" for value in SCALES])
            axis.set_yticks(
                range(len(correction_order)), [f"{value:.1f}" for value in correction_order]
            )
            axis.set_xlabel("Initial Prompt Scale")
            axis.set_ylabel("Correction Prompt Scale")
            axis.set_title(f"{LABELS[dataset]} — {title}", fontweight="bold")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Figure 3: Prompt-scale effect after {iteration} learned correction decision"
        + ("" if iteration == 1 else "s"),
        fontsize=14,
        fontweight="bold",
    )
    stem = f"figure3_after_iteration_{iteration}"
    fig.savefig(OUTPUT_ROOT / f"{stem}.png", dpi=300)
    fig.savefig(OUTPUT_ROOT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    write_csv(OUTPUT_ROOT / f"{stem}.csv", rows)


def plot_figure4(iteration: int) -> None:
    rows = []
    series = {}
    for dataset in ("mup", "sunseg", "vtus"):
        final_ious, deferred_rates, mean_corrections = [], [], []
        for alpha in ALPHAS:
            run_name = figure4_run(dataset, alpha)
            records = load_run(run_name, dataset)
            final_iou = float(
                np.mean([item["mean_iou_trajectory"][iteration] for item in records])
            )
            # First-decision deferral is the relevant workload measure when
            # comparing accept-initial-propagation versus request-one-correction.
            deferred = 100.0 * float(
                np.mean(
                    [
                        bool(item["decisions"])
                        and item["decisions"][0]["action"] == "correct"
                        for item in records
                    ]
                )
            )
            corrections = float(
                np.mean([item["correction_count_trajectory"][iteration] for item in records])
            )
            final_ious.append(final_iou)
            deferred_rates.append(deferred)
            mean_corrections.append(corrections)
            rows.append(
                {
                    "dataset": dataset,
                    "alpha": alpha,
                    "evaluation_iteration": iteration,
                    "mean_iou_after_selected_iteration": final_iou,
                    "first_decision_deferred_percent": deferred,
                    "mean_corrections_by_selected_iteration": corrections,
                    "run_name": run_name,
                }
            )
        series[dataset] = (final_ious, deferred_rates)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8), sharex=True, constrained_layout=True)
    for dataset in ("mup", "sunseg", "vtus"):
        final_ious, deferred_rates = series[dataset]
        axes[0].plot(
            ALPHAS,
            deferred_rates,
            marker="o",
            linewidth=2,
            color=COLORS[dataset],
            label=LABELS[dataset],
        )
        axes[1].plot(
            ALPHAS,
            final_ious,
            marker="o",
            linewidth=2,
            color=COLORS[dataset],
            label=LABELS[dataset],
        )
    axes[0].set_ylabel("Videos receiving ≥1 correction (%)")
    axes[0].set_ylim(-3, 103)
    axes[0].grid(alpha=0.25)
    axes[0].legend(title="Dataset")
    axes[1].set_ylabel("Final Mean IoU")
    axes[1].set_xlabel("Alpha (deferral cost)")
    axes[1].set_xticks(ALPHAS)
    axes[1].grid(alpha=0.25)
    axes[1].legend(title="Dataset")
    fig.suptitle(
        f"Figure 4: Deferral-cost effect after {iteration} learned correction decision"
        + ("" if iteration == 1 else "s"),
        fontsize=14,
        fontweight="bold",
    )
    stem = f"figure4_after_iteration_{iteration}"
    fig.savefig(OUTPUT_ROOT / f"{stem}.png", dpi=300)
    fig.savefig(OUTPUT_ROOT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    write_csv(OUTPUT_ROOT / f"{stem}.csv", rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iteration",
        type=int,
        choices=(1, 2, 3, 4),
        default=4,
        help="Plot the carried-forward cohort result after this many correction decisions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    plot_figure3(args.iteration)
    plot_figure4(args.iteration)
    print(f"Figure 3: {OUTPUT_ROOT / f'figure3_after_iteration_{args.iteration}.png'}")
    print(f"Figure 4: {OUTPUT_ROOT / f'figure4_after_iteration_{args.iteration}.png'}")


if __name__ == "__main__":
    main()
