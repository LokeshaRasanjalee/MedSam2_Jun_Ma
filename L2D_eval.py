import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

def build_eval_arg_parser() -> argparse.ArgumentParser:
    """
    Mirror L2D_train.py argument set as closely as possible, but add:
      - --eval_model_path (path to checkpoint .pth to load)
      - --save_csv_path (path to output CSV)

    Notes:
    - In training, `--output_mask_dir` and `--experiment_name` are required; for evaluation we keep them
      as optional to avoid forcing output directory creation.
    """
    parser = argparse.ArgumentParser()

    # --- Copied from L2D_train.py (same flag names/defaults) ---
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1_hiera_t512.yaml",
        help="MedSAM2  model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./checkpoints/MedSAM2_latest.pt",
        help="path to the MedSAM2 model checkpoint",
    )
    parser.add_argument(
        "-i",
        "--base_video_dir",
        type=str,
        help="directory containing videos (as JPEG files) to run inference on",
    )
    parser.add_argument(
        "-m",
        "--input_mask_dir",
        type=str,
        help="directory containing input masks (as PNG files) of each video",
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="text file containing the list of video names to run inference on",
    )
    parser.add_argument(
        "-o",
        "--output_mask_dir",
        type=str,
        required=False,
        default=None,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--use_all_masks",
        action="store_true",
        help="whether to use all available PNG files in input_mask_dir "
        "(default without this flag: just the first PNG file as input to the SAM 2 model; "
        "usually we don't need this flag, since semi-supervised VOS evaluation usually takes input from the first frame only)",
    )
    parser.add_argument(
        "--per_obj_png_file",
        action="store_true",
        help="whether use separate per-object PNG files for input and output masks "
        "(default without this flag: all object masks are packed into a single PNG file on each frame following DAVIS format; "
        "note that the SA-V dataset stores each object mask as an individual PNG file and requires this flag)",
    )
    parser.add_argument(
        "--save_palette_png",
        action="store_true",
        help="whether to save palette PNG files for output masks "
        "(default without this flag: all object masks are saved as grayscale PNG files (np.uint8) without palette)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    parser.add_argument(
        "--track_object_appearing_later_in_video",
        action="store_true",
        help="whether to track objects that appear later in the video (i.e. not on the first frame; "
        "some VOS datasets like LVOS or YouTube-VOS don't have all objects appearing in the first frame)",
    )
    parser.add_argument(
        "--use_vos_optimized_video_predictor",
        action="store_true",
        help="whether to use vos optimized video predictor with all modules compiled",
    )
    parser.add_argument(
        "-e",
        "--experiment_name",
        type=str,
        required=False,
        default="eval",
        help="Name of the experiment for logging and identification purposes",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate for training (default: 1e-2)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Alpha parameter for deferral loss (default: 1.0)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Beta parameter for deferral loss (default: 0.0)",
    )
    parser.add_argument(
        "--cost_tau",
        type=float,
        default=0.25,
        help="Softmax temperature (tau) for cost-sensitive weight computation in Mao losses (default: 0.25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--full_run",
        type=bool,
        default=True,
        help="Run the full training process (default: False)",
    )
    parser.add_argument(
        "--tensorboard_status",
        type=bool,
        default=False,
        help="tensorboard status (default: False)",
    )
    parser.add_argument(
        "--save_model",
        type=bool,
        default=False,
        help="Save model",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for training (default: 4)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0,
        help="Dropout rate for the model (default: 0.5)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save model every N epochs (default: 2)",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Load model path (default: None)",
    )
    parser.add_argument(
        "--wandb_status",
        type=bool,
        default=False,
        help="Wandb status (default: False)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=9,
        help="Number of classes for the model (default: 10)",
    )
    parser.add_argument(
        "--train_test_split",
        type=bool,
        default=False,
        help="Have seperate folders for train and test datasets(default: False)",
    )
    parser.add_argument(
        "--topk_values",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Top-k values to track for accuracy (default: [1, 3, 5])",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mae",
        help="Loss type (default: mae, log, exp)",
    )
    parser.add_argument(
        "--rgb_input",
        type=bool,
        default=True,
        help="RGB input (default: False)",
    )
    parser.add_argument(
        "--distance_type",
        type=str,
        default="quad",
        help="Distance type (default: exp, quad)",
    )
    parser.add_argument(
        "--distance_weight",
        type=float,
        default=0.4,
        help="Distance weight (default: 0.4)",
    )
    parser.add_argument(
        "--split_dict_path",
        type=str,
        default="./l2d_models/debug_run_mask_0/split_dict.txt",
        help="Split dict path (default: None)",
    )
    parser.add_argument(
        "--array_id",
        type=int,
        default=0,
        help="Array id (default: 0)",
    )
    parser.add_argument(
        "--data_npz_dir",
        type=str,
        default=None,
        help="Data npz dir (default: None)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sun",
        help="Dataset (default: sun, vtus, mup)",
    )

    # --- New eval-only args ---
    parser.add_argument(
        "--eval_model_path",
        type=str,
        default=None,
        help="Path to checkpoint (.pth) to load for evaluation. If omitted, will fall back to --load_model_path.",
    )
    parser.add_argument(
        "--save_csv_path",
        type=str,
        required=True,
        help="Path to save the output CSV.",
    )

    return parser


def _tensor_to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().view(-1)[0].item())


def _tensor_to_float_list(x: torch.Tensor) -> List[float]:
    arr = x.detach().cpu().view(-1).numpy().astype(np.float64)
    return [float(v) for v in arr.tolist()]


def main() -> None:
    parser = build_eval_arg_parser()
    args = parser.parse_args()

    # Lazy imports so `python L2D_eval.py --help` works even if heavy deps are missing.
    from dataloader import get_dataloaders
    from L2D_train import build_r2plus1d_model, set_seed

    set_seed(args.seed)

    # Determine checkpoint path (new arg preferred; fallback to training arg)
    ckpt_path = args.eval_model_path or args.load_model_path
    if not ckpt_path:
        raise ValueError("Must provide --eval_model_path (or --load_model_path as a fallback).")

    if args.data_npz_dir is None:
        raise ValueError("--data_npz_dir must be provided (directory containing .npz files).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build and load model
    model = build_r2plus1d_model(num_classes=args.num_classes, dropout_p=args.dropout, rgb_input=args.rgb_input)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Allow loading raw state_dict
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Reuse existing dataloader code; only test_loader is used here
    _, _, test_loader = get_dataloaders(args, batch_size=args.batch_size)
    if test_loader is None:
        raise ValueError(
            "test_loader is None. Ensure your split file contains key 2 (test) and that --train_test_split is False."
        )

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for clips_batch, no_df_batch, post_df_batch, npz_name_batch in test_loader:
            clips_batch = clips_batch.to(device)
            logits = model(clips_batch)  # [B, num_classes]

            if logits.dim() != 2:
                raise RuntimeError(f"Expected logits with shape [B, N], got shape {tuple(logits.shape)}")
            if logits.size(1) != 9:
                raise RuntimeError(
                    f"Expected 9 logits (for logit_1..logit_9). Got {logits.size(1)}. "
                    f"If this is intended, update the CSV schema."
                )

            logits_cpu = logits.detach().cpu()
            # Ensure shapes are [B] and [B, 9]
            no_df_batch = no_df_batch.view(-1)
            post_df_batch = post_df_batch.view(post_df_batch.size(0), -1)

            for i in range(logits_cpu.size(0)):
                logit_vals = logits_cpu[i].numpy().astype(np.float64).tolist()
                row: Dict[str, Any] = {
                    "npz_file_name": str(npz_name_batch[i]),
                    **{f"logit_{j}_value": float(logit_vals[j - 1]) for j in range(1, 10)},
                    "global_no_df_loss_complement": _tensor_to_float(no_df_batch[i]),
                    # store list as JSON string for safe round-tripping in CSV
                    "global_post_df_loss_complement": json.dumps(_tensor_to_float_list(post_df_batch[i])),
                }
                rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.save_csv_path)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.save_csv_path, index=False)
    print(f"Saved CSV with {len(df)} rows to: {args.save_csv_path}")


if __name__ == "__main__":
    main()


