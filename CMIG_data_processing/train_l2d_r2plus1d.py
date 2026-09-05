#!/usr/bin/env python3
"""Train a cost-aware R(2+1)D rejector on SUN, VTUS, or MUP intermediate data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import (
    R2Plus1D_18_Weights,
    R3D_18_Weights,
    r2plus1d_18,
    r3d_18,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DEFAULTS = {
    "sunseg": {
        "root": REPO_ROOT / "CMIG_l2d_data/sunseg",
        "prompt_dataset": "sunseg_14_10",
        "channels": 4,
        "video_key": "rgb_frames",
    },
    "vtus": {
        "root": REPO_ROOT / "CMIG_l2d_data/vtus",
        "prompt_dataset": "vtus_14_10",
        "channels": 2,
        "video_key": "grayscale_frames",
    },
    "mup": {
        "root": REPO_ROOT / "CMIG_l2d_data/mup",
        "prompt_dataset": "mup_mask_prompts",
        "channels": 2,
        "video_key": "grayscale_frames",
    },
}
SPLITS = ("train", "val", "test")
METRIC_DIRECTIONS = {
    "val_chosen_cost": "min",
    "val_chosen_iou": "max",
    "test_chosen_cost": "min",
    "test_chosen_iou": "max",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="sunseg")
    parser.add_argument(
        "--architecture",
        choices=("r2plus1d_18", "r2plus1d_18_temporal", "r3d_18", "resnet18_gru"),
        default="r2plus1d_18",
    )
    parser.add_argument("--data-root", type=Path, help="Override the dataset's CMIG_l2d_data root.")
    parser.add_argument(
        "--prompt-dataset",
        help="Intermediate subfolder. Default for SUN is sunseg_14_10 (initial 1.4, correction 1.0).",
    )
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "CMIG_l2d_training")
    parser.add_argument("--experiment-name", help="Default: <dataset>_<prompt-dataset>_<loss>.")
    parser.add_argument(
        "--loss",
        choices=("mae", "log", "exp", "mao_logistic"),
        default="mae",
        help=(
            "Surrogate loss. mao_logistic uses Mao et al.'s two-stage regression-deferral "
            "surrogate with 11 learned logits (stop plus ten frame actions)."
        ),
    )
    parser.add_argument("--beta", type=float, default=0.05, help="Cost added to every correction action.")
    parser.add_argument("--cost-tau", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-7)
    parser.add_argument(
        "--gru-learning-rate",
        type=float,
        help="Optional ResNet-GRU learning rate for the randomly initialized GRU; default uses --learning-rate.",
    )
    parser.add_argument(
        "--head-learning-rate",
        type=float,
        help="Optional learning rate for a randomly initialized shared score head; default uses --learning-rate.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.0)
    parser.add_argument("--gru-hidden-size", type=int, default=256)
    parser.add_argument(
        "--gru-unidirectional",
        action="store_true",
        help="Use a unidirectional GRU; the default ResNet-GRU is bidirectional.",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load Kinetics-400 R(2+1)D weights.")
    parser.add_argument("--pretrained-weights", type=Path, help="Optional local torchvision-compatible state_dict.")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = DATASET_DEFAULTS[args.dataset]
    args.data_root = (args.data_root or defaults["root"]).resolve()
    args.prompt_dataset = args.prompt_dataset or defaults["prompt_dataset"]
    args.input_channels = int(defaults["channels"])
    args.video_key = str(defaults["video_key"])
    args.output_root = args.output_root.resolve()
    args.experiment_name = args.experiment_name or (
        f"{args.dataset}_{args.prompt_dataset}_arch_{args.architecture}_{args.loss}"
    )
    if (
        args.epochs < 1
        or args.eval_every < 1
        or args.batch_size < 1
        or args.workers < 0
        or args.gru_hidden_size < 1
    ):
        raise ValueError("epochs, eval-every and batch-size must be positive; workers cannot be negative")
    if args.beta < 0 or args.cost_tau <= 0 or args.learning_rate <= 0:
        raise ValueError("beta must be nonnegative and cost-tau must be positive")
    if args.gru_learning_rate is not None and args.gru_learning_rate <= 0:
        raise ValueError("--gru-learning-rate must be positive")
    if args.head_learning_rate is not None and args.head_learning_rate <= 0:
        raise ValueError("--head-learning-rate must be positive")
    if args.architecture != "resnet18_gru" and args.gru_learning_rate is not None:
        raise ValueError("--gru-learning-rate requires --architecture resnet18_gru")
    if args.architecture not in {"resnet18_gru", "r2plus1d_18_temporal"} and args.head_learning_rate is not None:
        raise ValueError("--head-learning-rate requires a shared-head architecture")
    if not 0 <= args.horizontal_flip_prob <= 1:
        raise ValueError("horizontal-flip-prob must be in [0,1]")
    return args


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


class L2DIntermediateDataset(Dataset):
    def __init__(
        self,
        root: Path,
        prompt_dataset: str,
        split: str,
        video_key: str,
        input_channels: int,
        architecture: str,
        horizontal_flip_prob: float = 0.0,
    ) -> None:
        self.root = root
        self.prompt_dataset = prompt_dataset
        self.split = split
        self.video_key = video_key
        self.input_channels = input_channels
        self.architecture = architecture
        self.horizontal_flip_prob = horizontal_flip_prob
        sample_root = root / prompt_dataset / "samples"
        if not sample_root.is_dir():
            raise FileNotFoundError(f"Sample folder does not exist: {sample_root}")
        self.samples: list[Path] = []
        for path in sorted(sample_root.glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                if str(data["split"].item()) == split:
                    self.samples.append(path)
        if not self.samples:
            raise RuntimeError(f"No {split} samples found in {sample_root}")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    @lru_cache(maxsize=64)
    def _load_shared(path_string: str, video_key: str) -> tuple[np.ndarray, np.ndarray]:
        with np.load(path_string, allow_pickle=False) as data:
            frames = data[video_key].copy()
            locations = data["candidate_frame_indices"].copy()
        frames.setflags(write=False)
        locations.setflags(write=False)
        return frames, locations

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_path = self.samples[index]
        with np.load(sample_path, allow_pickle=False) as data:
            clip = str(data["clip_name"].item())
            masks = data["propagated_masks"].copy()
            action_ious = data["action_ious"].copy()
            selected = data["already_prompted_mask"].copy()
            locations = data["candidate_frame_indices"].copy()
            round_number = int(data["round"].item())
        shared_path = self.root / "shared_videos" / f"{clip}.npz"
        frames, shared_locations = self._load_shared(str(shared_path), self.video_key)
        if not np.array_equal(locations, shared_locations):
            raise ValueError(f"Candidate locations differ between {sample_path} and {shared_path}")
        expected_video_channels = self.input_channels - 1
        if frames.shape != (expected_video_channels, 10, 112, 112):
            raise ValueError(f"Unexpected video shape in {shared_path}: {frames.shape}")
        if masks.shape != (1, 10, 112, 112) or action_ious.shape != (11,) or selected.shape != (10,):
            raise ValueError(f"Unexpected intermediate shapes in {sample_path}")
        if not np.array_equal(np.isnan(action_ious[1:]), selected):
            raise ValueError(f"Invalid action availability encoding in {sample_path}")

        frame_tensor = torch.from_numpy(np.array(frames, copy=True)).float().div_(255.0)
        if expected_video_channels == 3:
            if self.architecture == "resnet18_gru":
                mean_values = (0.485, 0.456, 0.406)
                std_values = (0.229, 0.224, 0.225)
            else:
                mean_values = (0.43216, 0.394666, 0.37645)
                std_values = (0.22803, 0.22145, 0.216989)
            mean = torch.tensor(mean_values).view(3, 1, 1, 1)
            std = torch.tensor(std_values).view(3, 1, 1, 1)
            frame_tensor = (frame_tensor - mean) / std
        else:
            if self.architecture == "resnet18_gru":
                frame_tensor = (frame_tensor - 0.449) / 0.226
            else:
                frame_tensor = (frame_tensor - 0.400) / 0.225
        mask_tensor = torch.from_numpy(masks).float()
        model_input = torch.cat((frame_tensor, mask_tensor), dim=0)
        if self.horizontal_flip_prob and random.random() < self.horizontal_flip_prob:
            model_input = model_input.flip(-1)

        valid = np.concatenate((np.ones(1, dtype=np.bool_), ~selected))
        safe_ious = np.nan_to_num(action_ious, nan=0.0)
        return {
            "input": model_input,
            "action_ious": torch.from_numpy(safe_ious),
            "valid_action_mask": torch.from_numpy(valid),
            "candidate_frame_indices": torch.from_numpy(locations.astype(np.int64)),
            "round": round_number,
            "clip_name": clip,
        }


def make_loaders(args: argparse.Namespace) -> dict[str, DataLoader]:
    loaders = {}
    for split in SPLITS:
        dataset = L2DIntermediateDataset(
            root=args.data_root,
            prompt_dataset=args.prompt_dataset,
            split=split,
            video_key=args.video_key,
            input_channels=args.input_channels,
            architecture=args.architecture,
            horizontal_flip_prob=args.horizontal_flip_prob if split == "train" else 0.0,
        )
        generator = torch.Generator().manual_seed(args.seed)
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            drop_last=False,
            generator=generator,
        )
    return loaders


def replace_video_input_stem(model: nn.Module, input_channels: int) -> None:
    old: nn.Conv3d = model.stem[0]
    new = nn.Conv3d(
        input_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    with torch.no_grad():
        mean_weight = old.weight.mean(dim=1, keepdim=True)
        if input_channels == 4:
            new.weight[:, :3].copy_(old.weight)
            new.weight[:, 3:4].copy_(mean_weight)
        elif input_channels == 2:
            new.weight[:, 0:1].copy_(mean_weight)
            new.weight[:, 1:2].copy_(mean_weight)
        else:
            raise ValueError(f"Only 2- and 4-channel inputs are supported, got {input_channels}")
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.stem[0] = new


def load_local_state(model: nn.Module, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def replace_resnet_input_stem(model: nn.Module, input_channels: int) -> None:
    old: nn.Conv2d = model.conv1
    new = nn.Conv2d(
        input_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    with torch.no_grad():
        mean_weight = old.weight.mean(dim=1, keepdim=True)
        if input_channels == 4:
            new.weight[:, :3].copy_(old.weight)
            new.weight[:, 3:4].copy_(mean_weight)
        elif input_channels == 2:
            new.weight[:, 0:1].copy_(mean_weight)
            new.weight[:, 1:2].copy_(mean_weight)
        else:
            raise ValueError(f"Only 2- and 4-channel inputs are supported, got {input_channels}")
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.conv1 = new


class ResNet18GRU(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_size: int,
        bidirectional: bool,
        pretrained_weights: Path | None,
        no_pretrained: bool,
        learn_stop_logit: bool = False,
    ) -> None:
        super().__init__()
        if pretrained_weights:
            backbone = resnet18(weights=None)
            load_local_state(backbone, pretrained_weights)
            print(f"Loaded ResNet-18 weights from {pretrained_weights}")
        elif no_pretrained:
            backbone = resnet18(weights=None)
            print("Using randomly initialized ResNet-18 weights")
        else:
            try:
                backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            except Exception as error:
                raise RuntimeError(
                    "Could not load pretrained ImageNet ResNet-18 weights. Make them available in the "
                    "torch cache, provide --pretrained-weights, or use --no-pretrained."
                ) from error
            print("Loaded torchvision ImageNet-1K ResNet-18 weights")
        replace_resnet_input_stem(backbone, input_channels)
        backbone.fc = nn.Identity()
        for module in (backbone.conv1, backbone.bn1, backbone.layer1, backbone.layer2, backbone.layer3):
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.backbone = backbone
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        gru_features = hidden_size * (2 if bidirectional else 1)
        self.score_head = nn.Linear(gru_features, 1)
        nn.init.normal_(self.score_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.score_head.bias)
        self.stop_head = nn.Linear(gru_features, 1) if learn_stop_logit else None
        if self.stop_head is not None:
            nn.init.normal_(self.stop_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.stop_head.bias)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = video.shape
        frame_batch = video.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        features = self.backbone(frame_batch).reshape(batch, frames, 512)
        temporal_features, _ = self.gru(features)
        frame_logits = self.score_head(temporal_features).squeeze(-1)
        if self.stop_head is None:
            return frame_logits
        stop_logit = self.stop_head(temporal_features.mean(dim=1))
        return torch.cat((stop_logit, frame_logits), dim=1)


class TemporalR2Plus1D18(nn.Module):
    """R(2+1)D-18 retaining ten aligned temporal features with a shared scorer."""

    def __init__(
        self,
        backbone: nn.Module,
        expected_frames: int = 10,
        learn_stop_logit: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.expected_frames = expected_frames
        # Torchvision downsamples time in the first block of layers 2-4.
        # Preserve temporal length while retaining the original spatial stride.
        for layer_name in ("layer2", "layer3", "layer4"):
            first_block = getattr(self.backbone, layer_name)[0]
            temporal_conv: nn.Conv3d = first_block.conv1[0][3]
            temporal_conv.stride = (1, 1, 1)
            downsample_conv: nn.Conv3d = first_block.downsample[0]
            downsample_conv.stride = (1, 2, 2)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        for module_name in ("stem", "layer1", "layer2", "layer3"):
            for parameter in getattr(self.backbone, module_name).parameters():
                parameter.requires_grad = False
        self.score_head = nn.Linear(512, 1)
        nn.init.normal_(self.score_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.score_head.bias)
        self.stop_head = nn.Linear(512, 1) if learn_stop_logit else None
        if self.stop_head is not None:
            nn.init.normal_(self.stop_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.stop_head.bias)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        features = self.backbone.stem(video)
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        # Pool height/width only: [B,512,T,H,W] -> [B,T,512].
        temporal_features = features.mean(dim=(-1, -2)).transpose(1, 2)
        if temporal_features.shape[1] != self.expected_frames:
            raise RuntimeError(
                f"Temporal R(2+1)D produced {temporal_features.shape[1]} positions; "
                f"expected {self.expected_frames}"
            )
        frame_logits = self.score_head(temporal_features).squeeze(-1)
        if self.stop_head is None:
            return frame_logits
        stop_logit = self.stop_head(temporal_features.mean(dim=1))
        return torch.cat((stop_logit, frame_logits), dim=1)


def build_video_resnet(args: argparse.Namespace) -> nn.Module:
    if args.architecture == "r2plus1d_18":
        builder = r2plus1d_18
        default_weights = R2Plus1D_18_Weights.DEFAULT
        display_name = "R(2+1)D-18"
    elif args.architecture == "r3d_18":
        builder = r3d_18
        default_weights = R3D_18_Weights.DEFAULT
        display_name = "R3D-18"
    else:
        raise ValueError(f"Not a VideoResNet architecture: {args.architecture}")
    if args.pretrained_weights:
        model = builder(weights=None)
        load_local_state(model, args.pretrained_weights)
        print(f"Loaded {display_name} weights from {args.pretrained_weights}")
    elif args.no_pretrained:
        model = builder(weights=None)
        print(f"Using randomly initialized {display_name} weights")
    else:
        try:
            model = builder(weights=default_weights)
        except Exception as error:
            raise RuntimeError(
                f"Could not load pretrained Kinetics weights for {display_name}. Make them available "
                "in the torch cache, provide --pretrained-weights, or use --no-pretrained."
            ) from error
        print(f"Loaded torchvision Kinetics-400 {display_name} weights")
    replace_video_input_stem(model, args.input_channels)
    output_actions = 11 if args.loss == "mao_logistic" else 10
    model.fc = nn.Linear(model.fc.in_features, output_actions)
    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.fc.bias)
    # Match L2D_train.py: freeze the stem and residual stages 1-3, while
    # fine-tuning residual stage 4 and the replacement classifier.
    for module_name in ("stem", "layer1", "layer2", "layer3"):
        for parameter in getattr(model, module_name).parameters():
            parameter.requires_grad = False
    return model


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.architecture == "r2plus1d_18_temporal":
        # Build/load the ordinary pretrained R(2+1)D model first, then retain
        # temporal resolution and replace its classifier with a shared scorer.
        backbone_args = argparse.Namespace(**vars(args))
        backbone_args.architecture = "r2plus1d_18"
        backbone = build_video_resnet(backbone_args)
        # build_video_resnet installed a temporary fc and froze stages 1-3;
        # the wrapper replaces that fc and enforces the intended parameter set.
        return TemporalR2Plus1D18(
            backbone,
            learn_stop_logit=args.loss == "mao_logistic",
        )
    if args.architecture in {"r2plus1d_18", "r3d_18"}:
        return build_video_resnet(args)
    if args.architecture == "resnet18_gru":
        return ResNet18GRU(
            input_channels=args.input_channels,
            hidden_size=args.gru_hidden_size,
            bidirectional=not args.gru_unidirectional,
            pretrained_weights=args.pretrained_weights,
            no_pretrained=args.no_pretrained,
            learn_stop_logit=args.loss == "mao_logistic",
        )
    raise ValueError(f"Unsupported architecture: {args.architecture}")


def full_action_scores(frame_scores: torch.Tensor, valid: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mao_logistic":
        if frame_scores.shape[1] != 11:
            raise ValueError(
                f"Mao regression-deferral mode requires 11 learned logits, got {frame_scores.shape[1]}"
            )
        return frame_scores.masked_fill(~valid, -torch.inf)
    if frame_scores.shape[1] != 10:
        raise ValueError(f"Legacy loss mode requires ten learned frame scores, got {frame_scores.shape[1]}")
    stop = torch.zeros((frame_scores.shape[0], 1), dtype=frame_scores.dtype, device=frame_scores.device)
    scores = torch.cat((stop, frame_scores), dim=1)
    return scores.masked_fill(~valid, torch.inf)


def action_costs(action_ious: torch.Tensor, valid: torch.Tensor, beta: float) -> torch.Tensor:
    costs = 1.0 - action_ious
    costs[:, 1:] += beta
    return costs.masked_fill(~valid, torch.inf)


def cost_target_weights(costs: torch.Tensor, valid: torch.Tensor, tau: float) -> torch.Tensor:
    logits = (-costs / tau).masked_fill(~valid, -torch.inf)
    return torch.softmax(logits, dim=1).detach()


def surrogate_loss(scores: torch.Tensor, costs: torch.Tensor, valid: torch.Tensor, loss_name: str, tau: float) -> torch.Tensor:
    if loss_name == "mao_logistic":
        # Mao et al. (2024), Regression with Multi-Expert Deferral, Eq. (3).
        # Define C_0=L(h(x),y) for stop and C_j=c_j(x,y) for correction j.
        # The coefficient of multiclass loss ell(r,x,k) is sum(C)-C_k.
        # Invalid/already-prompted actions are omitted from the cost sum,
        # log-softmax denominator, and loss terms. No per-sample normalization
        # or cost-temperature transformation is applied.
        finite_costs = costs.masked_fill(~valid, 0.0)
        total_cost = finite_costs.sum(dim=1, keepdim=True)
        mao_weights = (total_cost - finite_costs).masked_fill(~valid, 0.0).detach()
        valid_logits = scores.masked_fill(~valid, -torch.inf)
        # The paper defines multiclass logistic loss with log base 2.
        negative_log_prob = -F.log_softmax(valid_logits, dim=1) / math.log(2.0)
        negative_log_prob = negative_log_prob.masked_fill(~valid, 0.0)
        return (mao_weights * negative_log_prob).sum(dim=1).mean()
    weights = cost_target_weights(costs, valid, tau)
    utilities = (-scores).masked_fill(~valid, -torch.inf)
    log_probs = F.log_softmax(utilities, dim=1)
    if loss_name == "log":
        return -(weights.masked_fill(~valid, 0.0) * log_probs.masked_fill(~valid, 0.0)).sum(1).mean()
    probabilities = torch.softmax(utilities, dim=1)
    if loss_name == "mae":
        return (1.0 - (weights * probabilities).sum(1)).mean()
    # Weighted multiclass exponential surrogate: for target i, sum_{k != i} exp(u_k-u_i).
    # Invalid target and competitor actions contribute exactly zero.
    pairwise = utilities.unsqueeze(1) - utilities.unsqueeze(2)  # [B,target_i,competitor_k] = u_k-u_i
    pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)
    identity = torch.eye(scores.shape[1], dtype=torch.bool, device=scores.device).unsqueeze(0)
    exp_terms = torch.exp(pairwise.clamp(min=-50.0, max=50.0)).masked_fill(~pair_valid | identity, 0.0)
    per_target = exp_terms.sum(dim=2)
    return (weights * per_target).sum(1).mean()


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums: defaultdict[str, float] = defaultdict(float)
        self.counts: defaultdict[str, int] = defaultdict(int)
        self.totals: defaultdict[str, float] = defaultdict(float)

    def add(self, key: str, values: torch.Tensor) -> None:
        values = values.detach().float()
        self.sums[key] += values.sum().item()
        self.counts[key] += values.numel()

    def add_scalar(self, key: str, value: float, count: int) -> None:
        self.sums[key] += value * count
        self.counts[key] += count

    def add_total(self, key: str, value: float) -> None:
        self.totals[key] += value

    def result(self) -> dict[str, float]:
        result = {key: self.sums[key] / self.counts[key] for key in self.sums if self.counts[key]}
        result.update(self.totals)
        return result


def update_decision_metrics(
    meter: MetricAccumulator,
    prefix: str,
    scores: torch.Tensor,
    costs: torch.Tensor,
    ious: torch.Tensor,
    valid: torch.Tensor,
    locations: torch.Tensor,
    higher_score_is_better: bool = False,
) -> None:
    batch = scores.shape[0]
    chosen = scores.argmax(1) if higher_score_is_better else scores.argmin(1)
    oracle = costs.argmin(1)
    chosen_iou = ious.gather(1, chosen[:, None]).squeeze(1)
    oracle_iou = ious.gather(1, oracle[:, None]).squeeze(1)
    chosen_cost = costs.gather(1, chosen[:, None]).squeeze(1)
    oracle_cost = costs.gather(1, oracle[:, None]).squeeze(1)
    best_iou = ious.masked_fill(~valid, -torch.inf).max(1).values
    meter.add(prefix + "exact_action_accuracy", (chosen == oracle).float())
    meter.add(prefix + "chosen_iou", chosen_iou)
    meter.add(prefix + "best_available_iou", best_iou)
    meter.add(prefix + "oracle_iou", oracle_iou)
    meter.add(prefix + "chosen_cost", chosen_cost)
    meter.add(prefix + "oracle_cost", oracle_cost)
    meter.add(prefix + "cost_regret", chosen_cost - oracle_cost)
    meter.add(prefix + "iou_regret", best_iou - chosen_iou)
    meter.add(prefix + "model_deferral_rate", (chosen > 0).float())
    meter.add(prefix + "oracle_deferral_rate", (oracle > 0).float())
    meter.add(prefix + "stop_defer_disagreement", ((chosen > 0) != (oracle > 0)).float())

    order_by_score = scores.argsort(1, descending=higher_score_is_better)
    for k in (1, 3, 5):
        meter.add(prefix + f"top{k}_accuracy", (order_by_score[:, :k] == oracle[:, None]).any(1).float())
    order_by_cost = costs.argsort(1)
    chosen_rank = (order_by_cost == chosen[:, None]).nonzero(as_tuple=False)[:, 1]
    meter.add(prefix + "chosen_action_cost_rank", chosen_rank.float())

    both_defer = (chosen > 0) & (oracle > 0)
    meter.add_scalar(prefix + "distance_eligible_rate", both_defer.float().sum().item() / batch, batch)
    meter.add_total(prefix + "distance_eligible_count", both_defer.sum().item())
    if both_defer.any():
        chosen_slot = chosen[both_defer] - 1
        oracle_slot = oracle[both_defer] - 1
        meter.add(prefix + "candidate_index_distance", (chosen_slot - oracle_slot).abs().float())
        chosen_frame = locations[both_defer].gather(1, chosen_slot[:, None]).squeeze(1)
        oracle_frame = locations[both_defer].gather(1, oracle_slot[:, None]).squeeze(1)
        meter.add(prefix + "temporal_frame_distance", (chosen_frame - oracle_frame).abs().float())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    meter = MetricAccumulator()
    amp_enabled = not args.no_amp
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        ious = batch["action_ious"].to(device, non_blocking=True)
        valid = batch["valid_action_mask"].to(device, non_blocking=True)
        locations = batch["candidate_frame_indices"].to(device, non_blocking=True)
        rounds = batch["round"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                frame_scores = model(inputs)
                scores = full_action_scores(frame_scores, valid, args.loss)
                costs = action_costs(ious, valid, args.beta)
                loss = surrogate_loss(scores, costs, valid, args.loss, args.cost_tau)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
        count = inputs.shape[0]
        meter.add_scalar("loss", loss.detach().item(), count)
        update_decision_metrics(
            meter,
            "",
            scores.detach(),
            costs,
            ious,
            valid,
            locations,
            higher_score_is_better=args.loss == "mao_logistic",
        )
        for round_number in range(1, 5):
            selected = rounds == round_number
            if selected.any():
                update_decision_metrics(
                    meter,
                    f"round_{round_number}/",
                    scores[selected].detach(),
                    costs[selected],
                    ious[selected],
                    valid[selected],
                    locations[selected],
                    higher_score_is_better=args.loss == "mao_logistic",
                )
    return meter.result()


def write_tensorboard(writer: SummaryWriter, split: str, metrics: dict[str, float], epoch: int) -> None:
    for name, value in metrics.items():
        if math.isfinite(value):
            writer.add_scalar(f"{split}/{name}", value, epoch)


def append_history(path: Path, epoch: int, split: str, metrics: dict[str, float]) -> None:
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "split", "metric", "value"))
        if not exists:
            writer.writeheader()
        for metric, value in sorted(metrics.items()):
            writer.writerow({"epoch": epoch, "split": split, "metric": metric, "value": value})


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best: dict[str, float],
    args: argparse.Namespace,
) -> None:
    temporary = path.with_name(f".{path.name}.writing-{os.getpid()}")
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best": best,
        "args": vars(args),
    }
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, dict[str, float]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1, dict(checkpoint.get("best", {}))


def print_cuda_diagnostics() -> None:
    print("CUDA/PyTorch diagnostics")
    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch compiled CUDA runtime: {torch.version.cuda}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for L2D model training; CPU fallback is disabled")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")


def main() -> None:
    args = resolve_args(parse_args())
    print_cuda_diagnostics()
    seed_everything(args.seed, args.deterministic)
    device = torch.device("cuda:0")
    run_dir = args.output_root / args.experiment_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"Dataset: {args.dataset}")
    print(f"Prompt dataset: {args.prompt_dataset}")
    print(f"Architecture: {args.architecture}")
    print(f"Input channels: {args.input_channels}")
    print(f"Loss: {args.loss}, beta={args.beta}, tau={args.cost_tau}")
    print(f"Output: {run_dir}")

    loaders = make_loaders(args)
    print("Samples: " + ", ".join(f"{split}={len(loaders[split].dataset)}" for split in SPLITS))
    model = build_model(args).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen_names = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if args.architecture == "resnet18_gru":
        expected_trainable_prefixes = ("backbone.layer4.", "gru.", "score_head.", "stop_head.")
        trainable_description = "ResNet layer4 + GRU + frame score head + optional stop head"
        frozen_description = "ResNet stem + layer1-layer3"
    elif args.architecture == "r2plus1d_18_temporal":
        expected_trainable_prefixes = ("backbone.layer4.", "score_head.", "stop_head.")
        trainable_description = "R(2+1)D layer4 + shared temporal frame head + optional stop head"
        frozen_description = "R(2+1)D stem + layer1-layer3"
    else:
        expected_trainable_prefixes = ("layer4.", "fc.")
        trainable_description = "layer4 + fc"
        frozen_description = "stem + layer1-layer3"
    if not trainable_names or any(
        not name.startswith(expected_trainable_prefixes) for name in trainable_names
    ):
        raise RuntimeError(f"Unexpected trainable parameter set: {trainable_names}")
    print(f"Trainable parameter tensors: {len(trainable_names)} ({trainable_description})")
    print(f"Frozen parameter tensors: {len(frozen_names)} ({frozen_description})")
    if args.architecture == "resnet18_gru" and (
        args.gru_learning_rate is not None or args.head_learning_rate is not None
    ):
        gru_lr = args.gru_learning_rate or args.learning_rate
        head_lr = args.head_learning_rate or args.learning_rate
        head_parameters = list(model.score_head.parameters())
        if model.stop_head is not None:
            head_parameters.extend(model.stop_head.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.layer4.parameters(), "lr": args.learning_rate, "group_name": "backbone_layer4"},
                {"params": model.gru.parameters(), "lr": gru_lr, "group_name": "gru"},
                {"params": head_parameters, "lr": head_lr, "group_name": "action_heads"},
            ],
            weight_decay=args.weight_decay,
        )
        print(
            "Optimizer learning rates: "
            f"backbone.layer4={args.learning_rate:g}, GRU={gru_lr:g}, score_head={head_lr:g}"
        )
    elif args.architecture == "r2plus1d_18_temporal" and args.head_learning_rate is not None:
        head_parameters = list(model.score_head.parameters())
        if model.stop_head is not None:
            head_parameters.extend(model.stop_head.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.layer4.parameters(), "lr": args.learning_rate, "group_name": "backbone_layer4"},
                {"params": head_parameters, "lr": args.head_learning_rate, "group_name": "action_heads"},
            ],
            weight_decay=args.weight_decay,
        )
        print(
            "Optimizer learning rates: "
            f"backbone.layer4={args.learning_rate:g}, score_head={args.head_learning_rate:g}"
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    start_epoch = 1
    best = {key: math.inf if direction == "min" else -math.inf for key, direction in METRIC_DIRECTIONS.items()}
    if args.resume:
        start_epoch, loaded_best = load_resume(args.resume, model, optimizer, scheduler, scaler, device)
        best.update(loaded_best)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        print(f"Received signal {signum}; a resumable checkpoint will be saved after this epoch.")
        stop_requested = True

    signal.signal(signal.SIGUSR2, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    # Keep every future experiment under one shared TensorBoard root so all
    # runs can be compared in a single interface. The named child directory
    # gives each run an unambiguous visible label.
    tensorboard_run_dir = args.output_root / "tensorboard" / args.experiment_name
    writer = SummaryWriter(tensorboard_run_dir)
    writer.add_text("run/experiment_name", args.experiment_name, 0)
    writer.add_text("run/dataset", args.dataset, 0)
    writer.add_text("run/prompt_dataset", args.prompt_dataset, 0)
    history_path = run_dir / "metrics.csv"

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            torch.cuda.reset_peak_memory_stats(device)
            train_metrics = run_epoch(model, loaders["train"], device, args, optimizer, scaler)
            train_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            for group in optimizer.param_groups:
                if "group_name" in group:
                    train_metrics[f"learning_rate/{group['group_name']}"] = group["lr"]
            train_metrics["epoch_seconds"] = time.time() - epoch_start
            train_metrics["peak_cuda_memory_gib"] = torch.cuda.max_memory_allocated(device) / (1024**3)
            write_tensorboard(writer, "train", train_metrics, epoch)
            append_history(history_path, epoch, "train", train_metrics)
            print(
                f"Epoch {epoch}/{args.epochs} train loss={train_metrics['loss']:.6f} "
                f"chosen_iou={train_metrics['chosen_iou']:.4f} chosen_cost={train_metrics['chosen_cost']:.4f}"
            )

            evaluate = epoch % args.eval_every == 0 or epoch == args.epochs
            if evaluate:
                for split in ("val", "test"):
                    evaluation_start = time.time()
                    metrics = run_epoch(model, loaders[split], device, args, None, scaler)
                    metrics["evaluation_seconds"] = time.time() - evaluation_start
                    write_tensorboard(writer, split, metrics, epoch)
                    append_history(history_path, epoch, split, metrics)
                    print(
                        f"  {split}: loss={metrics['loss']:.6f} chosen_iou={metrics['chosen_iou']:.4f} "
                        f"chosen_cost={metrics['chosen_cost']:.4f} regret={metrics['cost_regret']:.4f}"
                    )
                    for metric_suffix, metric_name in (("chosen_cost", "chosen_cost"), ("chosen_iou", "chosen_iou")):
                        key = f"{split}_{metric_suffix}"
                        value = metrics[metric_name]
                        direction = METRIC_DIRECTIONS[key]
                        improved = value < best[key] if direction == "min" else value > best[key]
                        if improved:
                            best[key] = value
                            save_checkpoint(
                                checkpoint_dir / f"best_{key}.pt",
                                model, optimizer, scheduler, scaler, epoch, best, args,
                            )
            scheduler.step()
            save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, scheduler, scaler, epoch, best, args)
            writer.flush()
            if stop_requested:
                print("Stopping after saving latest.pt")
                break
    finally:
        writer.close()


if __name__ == "__main__":
    main()
