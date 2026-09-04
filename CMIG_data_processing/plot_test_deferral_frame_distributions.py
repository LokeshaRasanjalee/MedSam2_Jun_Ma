#!/usr/bin/env python3
"""Plot model-versus-oracle selected-frame distributions on the non-iterative test set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_l2d_r2plus1d as trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "CMIG_l2d_training" / "deferral_frame_distributions")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def dataset_label(dataset: str) -> str:
    return {"sunseg": "SUNSEG", "vtus": "VTUS", "mup": "MUP"}[dataset]


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        config = json.loads((run_dir / "config.json").read_text())
        config["data_root"] = Path(config["data_root"])
        config["output_root"] = Path(config["output_root"])
        config["pretrained_weights"] = Path(config["pretrained_weights"]) if config.get("pretrained_weights") else None
        config["no_pretrained"] = True
        model_args = SimpleNamespace(**config)
        dataset = trainer.L2DIntermediateDataset(
            model_args.data_root, model_args.prompt_dataset, "test", model_args.video_key,
            model_args.input_channels, model_args.architecture,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        checkpoint_path = run_dir / "checkpoints" / "best_test_chosen_cost.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = trainer.build_model(model_args)
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()

        counts = {(round_number, source): Counter() for round_number in range(1, 5) for source in ("model", "oracle")}
        with torch.inference_mode():
            for batch in loader:
                valid = batch["valid_action_mask"].to(device)
                ious = batch["action_ious"].to(device)
                locations = batch["candidate_frame_indices"]
                model_action = trainer.full_action_scores(model(batch["input"].to(device)), valid).argmin(1).cpu()
                oracle_action = trainer.action_costs(ious, valid, model_args.beta).argmin(1).cpu()
                for index in range(len(model_action)):
                    round_number = int(batch["round"][index])
                    for source, action in (("model", int(model_action[index])), ("oracle", int(oracle_action[index]))):
                        if action > 0:
                            frame_index = int(locations[index, action - 1])
                            counts[(round_number, source)][frame_index] += 1

        csv_path = args.output_dir / f"{model_args.dataset}_test_deferral_frame_counts.csv"
        all_frames = sorted({frame for counter in counts.values() for frame in counter})
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dataset", "checkpoint_epoch", "round", "source", "frame_index", "count"])
            for round_number in range(1, 5):
                for source in ("model", "oracle"):
                    for frame in all_frames:
                        writer.writerow([model_args.dataset, checkpoint["epoch"], round_number, source, frame, counts[(round_number, source)][frame]])

        figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        if model_args.dataset == "sunseg":
            frame_domain = list(range(0, 100))
        elif model_args.dataset == "vtus":
            frame_domain = list(range(0, 30))
        else:
            frame_domain = list(range(0, 10))
        width = 0.42
        for round_number, axis in enumerate(axes.flat, start=1):
            model_values = [counts[(round_number, "model")][frame] for frame in frame_domain]
            oracle_values = [counts[(round_number, "oracle")][frame] for frame in frame_domain]
            axis.bar([x - width / 2 for x in frame_domain], model_values, width, label="Model", color="#377eb8")
            axis.bar([x + width / 2 for x in frame_domain], oracle_values, width, label="Oracle", color="#e41a1c", alpha=0.8)
            axis.set_title(
                f"Round {round_number}: model n={sum(model_values)}, oracle n={sum(oracle_values)}"
            )
            axis.set_ylabel("Deferral selections")
            axis.grid(axis="y", alpha=0.25)
        axes[1, 0].set_xlabel("Original frame index")
        axes[1, 1].set_xlabel("Original frame index")
        axes[0, 0].legend()
        figure.suptitle(
            f"{dataset_label(model_args.dataset)} test-set selected frames | lowest-test-cost checkpoint, epoch {checkpoint['epoch']}"
        )
        figure.tight_layout()
        output_path = args.output_dir / f"{model_args.dataset}_test_deferral_frame_distribution.png"
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        print(f"{dataset_label(model_args.dataset)}: {output_path}")
        print(f"  counts: {csv_path}")


if __name__ == "__main__":
    main()
