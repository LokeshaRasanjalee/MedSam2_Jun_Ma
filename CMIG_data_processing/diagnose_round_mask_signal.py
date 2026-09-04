#!/usr/bin/env python3
"""Diagnose whether propagated-mask changes across rounds alter rejector logits and decisions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

import train_l2d_r2plus1d as trainer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO_ROOT / "CMIG_l2d_training" / (
    "mup_mask_prompts_arch_r2plus1d_18_alpha_010_loss_mae_oracle_bal50_seed_42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, help="Default: <run-dir>/checkpoints/best_test_chosen_cost.pt")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def binary_overlap(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
    first = first.astype(bool)
    second = second.astype(bool)
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    total = first.sum() + second.sum()
    iou = float(intersection / union) if union else 1.0
    dice = float(2 * intersection / total) if total else 1.0
    changed = float(np.not_equal(first, second).mean())
    return iou, dice, changed


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_path = (args.checkpoint or run_dir / "checkpoints" / "best_test_chosen_cost.pt").resolve()
    config = json.loads((run_dir / "config.json").read_text())
    config["data_root"] = Path(config["data_root"])
    config["output_root"] = Path(config["output_root"])
    config["pretrained_weights"] = Path(config["pretrained_weights"]) if config.get("pretrained_weights") else None
    # The checkpoint replaces all weights, so avoid an unnecessary pretrained-weight download.
    config["no_pretrained"] = True
    model_args = SimpleNamespace(**config)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    dataset = trainer.L2DIntermediateDataset(
        root=model_args.data_root,
        prompt_dataset=model_args.prompt_dataset,
        split=args.split,
        video_key=model_args.video_key,
        input_channels=model_args.input_channels,
        architecture=model_args.architecture,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = trainer.build_model(model_args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    inference: dict[tuple[str, int], dict] = {}
    with torch.inference_mode():
        for batch in loader:
            raw_logits = model(batch["input"].to(device)).cpu()
            valid = batch["valid_action_mask"]
            scores = trainer.full_action_scores(raw_logits, valid)
            choices = scores.argmin(1)
            for row_index, clip in enumerate(batch["clip_name"]):
                round_number = int(batch["round"][row_index])
                action = int(choices[row_index])
                inference[(clip, round_number)] = {
                    "raw_logits": raw_logits[row_index].tolist(),
                    "valid": valid[row_index].tolist(),
                    "action": action,
                    "decision": "stop" if action == 0 else "defer",
                    "selected_candidate_slot": None if action == 0 else action - 1,
                    "selected_frame_index": None if action == 0 else int(batch["candidate_frame_indices"][row_index, action - 1]),
                }

    by_clip: dict[str, list[tuple[int, Path]]] = {}
    for path in dataset.samples:
        with np.load(path, allow_pickle=False) as data:
            by_clip.setdefault(str(data["clip_name"].item()), []).append((int(data["round"].item()), path))

    round_rows: list[dict] = []
    transition_rows: list[dict] = []
    for clip, entries in sorted(by_clip.items()):
        entries.sort()
        previous = None
        for round_number, path in entries:
            with np.load(path, allow_pickle=False) as data:
                mask = data["propagated_masks"].astype(bool)
                selected = data["already_prompted_mask"].astype(bool)
                action_ious = data["action_ious"].astype(float)
            result = inference[(clip, round_number)]
            valid_costs = 1.0 - action_ious
            valid_costs[1:] += float(model_args.beta)
            valid_costs[np.isnan(action_ious)] = np.inf
            oracle_action = int(np.argmin(valid_costs))
            row = {
                "clip_name": clip, "round": round_number,
                "prompted_slots": "|".join(map(str, np.flatnonzero(selected).tolist())),
                "current_iou": action_ious[0], "oracle_action": oracle_action,
                "oracle_decision": "stop" if oracle_action == 0 else "defer",
                "model_action": result["action"], "model_decision": result["decision"],
                "selected_candidate_slot": result["selected_candidate_slot"],
                "selected_frame_index": result["selected_frame_index"],
            }
            row.update({f"logit_frame_{index}": value for index, value in enumerate(result["raw_logits"])})
            round_rows.append(row)
            if previous is not None:
                previous_round, previous_mask, previous_result = previous
                iou, dice, changed = binary_overlap(previous_mask, mask)
                first_logits = np.asarray(previous_result["raw_logits"])
                second_logits = np.asarray(result["raw_logits"])
                transition_rows.append({
                    "clip_name": clip, "from_round": previous_round, "to_round": round_number,
                    "mask_iou": iou, "mask_dice": dice, "mask_changed_fraction": changed,
                    "logit_mean_abs_change": float(np.abs(second_logits - first_logits).mean()),
                    "logit_max_abs_change": float(np.abs(second_logits - first_logits).max()),
                    "decision_changed": int(previous_result["action"] != result["action"]),
                    "stop_defer_changed": int(previous_result["decision"] != result["decision"]),
                })
            previous = (round_number, mask, result)

    output_dir = (args.output_dir or run_dir / "round_mask_signal_diagnostic" / args.split).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_round.csv", round_rows)
    write_csv(output_dir / "round_transitions.csv", transition_rows)

    summary = {
        "run_dir": str(run_dir), "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)), "split": args.split,
        "clips": len(by_clip), "round_samples": len(round_rows), "transitions": len(transition_rows),
        "mean_mask_iou": float(np.mean([r["mask_iou"] for r in transition_rows])),
        "mean_mask_dice": float(np.mean([r["mask_dice"] for r in transition_rows])),
        "mean_mask_changed_fraction": float(np.mean([r["mask_changed_fraction"] for r in transition_rows])),
        "mean_logit_abs_change": float(np.mean([r["logit_mean_abs_change"] for r in transition_rows])),
        "action_change_rate": float(np.mean([r["decision_changed"] for r in transition_rows])),
        "stop_defer_change_rate": float(np.mean([r["stop_defer_changed"] for r in transition_rows])),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter([r["mask_changed_fraction"] for r in transition_rows], [r["logit_mean_abs_change"] for r in transition_rows], alpha=.7)
    axes[0].set(xlabel="Mask pixels changed between rounds", ylabel="Mean absolute logit change", title="Mask change vs rejector response")
    labels = [f"{a}→{b}" for a, b in ((1, 2), (2, 3), (3, 4))]
    values = [[r["logit_mean_abs_change"] for r in transition_rows if r["from_round"] == a and r["to_round"] == b] for a, b in ((1, 2), (2, 3), (3, 4))]
    axes[1].boxplot(values, tick_labels=labels)
    axes[1].set(xlabel="Round transition", ylabel="Mean absolute logit change", title="Rejector change by transition")
    figure.tight_layout()
    figure.savefig(output_dir / "mask_vs_logit_change.png", dpi=180)
    plt.close(figure)

    print(json.dumps(summary, indent=2))
    print(f"Saved diagnostic to: {output_dir}")


if __name__ == "__main__":
    main()
