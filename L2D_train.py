# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from collections import defaultdict
import datetime
import subprocess
import random
import time  # Add time module import
import json
from collections import deque
import glob
from sam2.build_sam import build_sam2_video_predictor
import wandb

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import logging
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import torchvision.models.video as models
from PIL import Image
from dataloader import get_dataloaders
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from collections import Counter
from cnn_3d import Simple3DCNN
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import psutil

def check_cuda():
    """Check if CUDA is available and print device information."""
    if torch.cuda.is_available():
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("CUDA is not available. Using CPU.")
        return False

# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette and confidence value."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    
    if mask.ndim == 3:
    # RGB mask → binary mask
        mask = np.any(mask != 0, axis=-1)
    elif mask.ndim == 2:
        # Already 2D, just ensure it's boolean
        mask = mask != 0
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")  
    object_ids = np.unique(mask).astype(int)
    object_ids = object_ids[object_ids > 0].tolist()
    
    pixel_counts = {}
    for obj_id in object_ids:
        pixel_counts[obj_id] = np.sum(mask == obj_id)
    
    # print(pixel_counts)
    
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette


def save_palette_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    output_palette,
    confidence_scores,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    confidence_scores,
):
    """Save masks to a directory as greyscale PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        assert output_mask.dtype == np.uint8
        assert output_mask.ndim == 2
        
        # Convert to binary mask (0 or 255) for better visibility
        output_mask = (output_mask > 0).astype(np.uint8) * 255
        output_mask = Image.fromarray(output_mask)
        output_mask.save(output_mask_path)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            assert output_mask.dtype == np.uint8
            assert output_mask.ndim == 2
            
            # Convert to binary mask (0 or 255) for better visibility
            output_mask = (output_mask > 0).astype(np.uint8) * 255
            output_mask = Image.fromarray(output_mask)
            output_mask.save(output_mask_path)
            
def get_frame_names(video_dir):
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names = list(sorted(frame_names))
    return frame_names  

def get_mask_img_list_with_obj(args, frame_names, video_name):
    mask_img_list = [
        name
        for idx, name in enumerate(frame_names)
        if os.path.exists(
            os.path.join(args.input_mask_dir, video_name, f"{name}.png")
        )
    ]
    mask_img_list_with_obj = sorted([
        idx
        for idx, name in enumerate(mask_img_list)
        if np.any(np.array(Image.open(os.path.join(args.input_mask_dir, video_name, f"{name}.png")).convert('L')) > 0)
    ])

    return mask_img_list_with_obj



def save_model(model, output_path, model_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    joblib.dump(model, os.path.join(output_path, model_name))
 
    


def compute_costs(acc_post_def_batch, alpha, beta):
    """
    batch_expert_preds: [batch, n_experts]
    y_true: [batch]
    Returns: costs [batch, n_experts]
    """
    incorrect = (1-acc_post_def_batch)
    cost = alpha * incorrect + beta  # shape [batch, n_experts]
    return cost


def deferral_loss(acc_no_def_batch, rejector_logits, acc_post_def_batch, alpha=1.0, beta=1.0):
    """
    Surrogate deferral loss adapted from Mao et al. (2023), L_exp in predictor-rejector setting.

    Args:
        acc_no_def_batch: [B] - segmentation accuracy from base model (no deferral), like 1_{h(x) == y}
        rejector_logits:  [B, n_e] - logits from rejector model (one per candidate frame)
        acc_post_def_batch: [B, n_e] - accuracy if frame j is chosen (complement of cost)
        alpha, beta: weights to balance no-deferral and deferral components

    Returns:
        scalar loss
    """
    B, n_e = rejector_logits.shape
    #rejector_logits = torch.clamp(rejector_logits, min=-10, max=10) # extra addition my me


    # First term: alpha * acc_no_def_batch * sum_i e^{-r_i}
    exp_neg_r = torch.exp(-rejector_logits)           # [B, n_e]
    term1 = torch.sum(exp_neg_r, dim=1)               # [B]
    loss_term1 = acc_no_def_batch * term1     # [B]

    # Second term: beta * sum_j acc_post_def * [sum_{i≠j} e^{r_j - r_i} + e^{r_j}]
    loss_term2 = torch.zeros_like(loss_term1)         # [B]
    for j in range(n_e):
        r_j = rejector_logits[:, j].unsqueeze(1)      # [B, 1]
        r_diff = r_j - rejector_logits                # [B, n_e]
        mask = torch.ones(n_e, dtype=torch.bool, device=rejector_logits.device)
        mask[j] = False
        exp_diff = torch.sum(torch.exp(r_diff[:, mask]), dim=1)  # [B]
        exp_rj = torch.exp(r_j.squeeze(1))            # [B]
        penalty = exp_diff + (n_e-1)*exp_rj                   # [B]

        # Cost calculation with clamping to prevent negative losses
        # cost = torch.clamp(alpha * (1 - acc_post_def_batch[:, j]) + beta, max=1.0)
        cost = alpha * (1 - acc_post_def_batch[:, j]) + beta
        c_bar = 1 - cost  # This will now always be >= 0

        loss_term2 += c_bar * penalty  # [B]

    # Combine both terms
    total_loss = loss_term1 + loss_term2              # [B]
    
    # print("=== DEBUG LOGS ===")
    # print("Rejector logits:\n", rejector_logits[:3])
    # print("acc_no_def_batch:\n", acc_no_def_batch[:3])
    # print("acc_post_def_batch:\n", acc_post_def_batch[:3])
    # print("loss_term1:\n", loss_term1[:3])
    # print("loss_term2:\n", loss_term2[:3])
    # print("total_loss:\n", total_loss[:3])
    
    return torch.mean(total_loss)

def onetime_deferal_loss(acc_no_def_batch, rejector_logits, acc_post_def_batch, beta, distance_loss):
    """
    Cost-sensitive cross-entropy loss for learning to defer.
    
    Parameters:
    - acc_no_def_batch: [B] — Dice scores for predictions using only prompt at f₀
    - rejector_logits:  [B, J+1] — model logits for deferral decisions (0 = no deferral, 1..J = frame-specific deferral)
    - acc_post_def_batch: [B, J] — Dice scores using f₀ + fⱼ, for j=1..J
    - alpha: scalar — constant deferral cost to be added to each post-deferral option
    
    Returns:
    - scalar loss (mean over batch)
    """
    B, num_classes = rejector_logits.shape  # num_classes = J + 1
    J = num_classes - 1
    
    # 1. Compute total cost vector: c(i) = 1 - acc(i) + lambda (only for deferred)
    c0 = 1.0 - acc_no_def_batch                         # [B] for no deferral
    c_defer = 1.0 - acc_post_def_batch + beta + distance_loss       # [B, J] with deferral cost added

    # 2. Combine into full cost matrix [B, J+1]
    cost = torch.cat([c0.unsqueeze(1), c_defer], dim=1)  # [B, J+1]

    # 3. Compute softmax weights: w(i) = max(c) - c(i)
    max_c, _ = cost.max(dim=1, keepdim=True)            # [B, 1]
    weights = max_c - cost                              # [B, J+1]

    # 4. Cross-entropy with custom weights
    log_probs = F.log_softmax(rejector_logits, dim=1)   # [B, J+1]
    loss = -torch.sum(weights * log_probs, dim=1)       # [B]
    return loss.mean()

def mao_regression_mae_loss(acc_no_def_batch, rejector_logits, acc_post_def_batch,beta, distance_loss):
    """
    Hierarchical deferral loss L^h_ℓ from the equation provided.

    Args:
        acc_no_def_batch: [B] — Dice score (or 1 - loss) using only h(x) (i.e., no deferral)
        rejector_logits:  [B, J+1] — logits over [no deferral, expert 1..J]
        acc_post_def_batch: [B, J] — Dice scores using expert j (j=1..J)
        distance_loss: [B, J] — additional cost for each deferral option (e.g., temporal distance)
    
    Returns:
        Scalar loss averaged over batch
    """
    B, num_actions = rejector_logits.shape
    n_e = num_actions - 1  # first index is no deferral (0)
        
    exp_logits = torch.exp(rejector_logits)             # [B, J+1]
    denominator = torch.sum(exp_logits, dim=1, keepdim=True)  # [B, 1]
    log_probs = 1 - (exp_logits / denominator)            # [B, J+1]
    
    #Term 1

    # Step 1: compute c_j(x, y) = 1 - acc + distance_loss for j=1..J
    cj = (1.0 - acc_post_def_batch) + beta + distance_loss  # [B, J]
    cj_sum_1 = torch.sum(cj, dim=1, keepdim=True)        # [B, 1] — sum over all expert costs for no deferral path

    # Step 2: loss for no deferral (r(x, 0)) — uses c0 as weight
    loss_term1 = cj_sum_1.squeeze(1) * log_probs[:, 0]  # [B]
    
    
    #Term 2
    loss_term2 = torch.zeros_like(loss_term1)         # [B]
    for j in range(n_e):
        l_j = log_probs[:, j+1]      # [B, 1]
        mask = torch.ones(n_e, dtype=torch.bool, device=rejector_logits.device)
        mask[j] = False
        cj_sum = torch.sum(cj[:, mask], dim=1) # [B]
        loss_term2 += ((cj_sum + (1-acc_no_def_batch)) * l_j) # [B]
                   
    # Combine both terms
    total_loss = loss_term1 + loss_term2    

    return total_loss.mean()
    

def train_one_epoch(rejector,epoch, loader, criterion, optimizer,save_every, alpha, beta, device, topk_values=[1, 3, 5], distance_loss=10):
    rejector.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    total_regret = 0.0

    all_best_actions = []
    all_chosen_actions = []
    all_video_names = []  # Add list to collect video names
    rank_distances = []
    total_chosen_acc = 0.0
    total_best_acc = 0.0
    
    # Add top-k accuracy tracking
    total_topk_correct = {k: 0 for k in topk_values}

    for clips_batch, no_df_dice_batch, post_df_dice_batch, video_name_batch in loader:
        clips_batch = clips_batch.to(device)
        no_df_dice_batch = no_df_dice_batch.to(device)
        post_df_dice_batch = post_df_dice_batch.to(device)
        
        optimizer.zero_grad()
        
        #input= clips_batch.permute(0, 2, 1, 3, 4)
        rej_logits = rejector(clips_batch)
        
        loss = mao_regression_mae_loss(no_df_dice_batch, rej_logits, post_df_dice_batch, beta, distance_loss)
   
        # Backward pass
        loss.backward()
        
        # for name, param in rejector.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} has gradient with mean  {param.grad.abs().mean()}")
        #     else:
        #         print(f"{name} has no gradient")
        
        optimizer.step()
        
        if (epoch+1) % save_every == 0:        
            total_loss += loss.item()

            # Calculate accuracy metrics
            chosen_actions = infer_deferral_action(rej_logits)
            all_accs = torch.cat([no_df_dice_batch.unsqueeze(1), post_df_dice_batch], dim=1)
            # Accuracy from chosen action
            chosen_accs = torch.gather(all_accs, 1, chosen_actions.unsqueeze(1)).squeeze(1)


            # Calculate adjusted gain by subtracting beta from post_df_dice_batch
            adjusted_gain = post_df_dice_batch - beta - distance_loss
            # All possible accuracies: base + n_e frames with adjusted gain
            all_accs_adjusted = torch.cat([no_df_dice_batch.unsqueeze(1), adjusted_gain], dim=1)
            # Best accuracy (oracle) using argmax on adjusted gains
            best_actions = torch.argmax(all_accs_adjusted, dim=1)
            best_accs = torch.gather(all_accs, 1, best_actions.unsqueeze(1)).squeeze(1)

            # Compute metrics
            correct += (chosen_actions == best_actions).sum().item()
            regret = torch.abs(best_accs - chosen_accs)
            total_regret += regret.sum().item()
            total_samples += clips_batch.size(0)
            total_chosen_acc += chosen_accs.sum().item()
            total_best_acc += best_accs.sum().item()
            
            # Compute top-k accuracy
            topk_accuracies = calculate_topk_accuracy(rej_logits, best_actions, topk_values)
            for k, acc in topk_accuracies.items():
                total_topk_correct[k] += acc * clips_batch.size(0)
            
            # Compute rank distance per sample in batch
            for i in range(all_accs.size(0)):
                accs = all_accs[i]
                chosen_idx = chosen_actions[i].item()
                sorted_indices = torch.argsort(accs, descending=True)
                rank = (sorted_indices == chosen_idx).nonzero(as_tuple=True)[0].item()
                rank_distances.append(rank)

            # Store results
            all_best_actions.append(best_actions.cpu())
            all_chosen_actions.append(chosen_actions.cpu())
            all_video_names.extend(video_name_batch)  # Collect video names
    
    if (epoch+1) % save_every == 0:
        avg_loss = total_loss / len(loader)
        selection_accuracy = correct / total_samples
        mean_regret = total_regret / total_samples
        avg_rank_distance = sum(rank_distances) / len(rank_distances)
        all_best_actions = torch.cat(all_best_actions)
        all_chosen_actions = torch.cat(all_chosen_actions)
        avg_chosen_acc = total_chosen_acc / total_samples
        avg_best_acc = total_best_acc / total_samples
        
        # Calculate top-k accuracies
        topk_accuracies = {k: total_topk_correct[k] / total_samples for k in topk_values}

        return avg_loss, selection_accuracy, mean_regret, all_best_actions, all_chosen_actions, avg_rank_distance, avg_chosen_acc, avg_best_acc, topk_accuracies, all_video_names
    else:
        return None, None, None, None, None, None, None, None, None, None

def infer_deferral_action(rejector_logits):
    """
    Given logits over deferral actions, return the index of the inferred best action.

    Parameters:
    - rejector_logits: Tensor of shape [B, N] where:
        - B = batch size
        - N = number of deferral options (e.g., 0 = no deferral, 1..J = frame-specific prompts)

    Returns:
    - action_idx: Tensor of shape [B] — index of the best action per sample
    """
    action_idx = torch.argmax(rejector_logits, dim=1)  # shape [B]
    return action_idx


def calculate_topk_accuracy(rejector_logits, best_actions, k_values=[1, 3, 5]):
    """
    Calculate top-k accuracy for deferral action prediction.
    
    Parameters:
    - rejector_logits: Tensor of shape [B, N] — model logits for each action
    - best_actions: Tensor of shape [B] — ground truth best actions
    - k_values: list of ints — top-k values to consider (default: [1, 3, 5])
    
    Returns:
    - topk_accuracies: dict — dictionary with k as key and accuracy as value
    """
    topk_accuracies = {}
    
    for k in k_values:
        if k == 1:
            # For k=1, it's the same as regular accuracy
            chosen_actions = torch.argmax(rejector_logits, dim=1)
            correct = (chosen_actions == best_actions).float().mean().item()
            topk_accuracies[k] = correct
        else:
            # Get top-k predicted actions
            topk_values, topk_indices = torch.topk(rejector_logits, k, dim=1)  # [B, k]
            
            # Check if the correct action is in the top-k predictions
            # Expand best_actions to match topk_indices shape for comparison
            best_actions_expanded = best_actions.unsqueeze(1).expand_as(topk_indices)  # [B, k]
            
            # Check if any of the top-k predictions match the correct action
            correct_in_topk = torch.any(topk_indices == best_actions_expanded, dim=1)  # [B]
            
            # Calculate accuracy
            topk_accuracies[k] = correct_in_topk.float().mean().item()
    
    return topk_accuracies

def validate_one_epoch(model, epoch, loader, criterion, alpha, beta, device, logging=None, topk_values=[1, 3, 5], distance_loss=10):
    model.eval()
    total_samples = 0
    total_regret = 0.0
    correct = 0
    total_val_loss = 0.0

    all_best_actions = []
    all_chosen_actions = []
    all_video_names = []  # Add list to collect video names
    rank_distances = []  # <-- new list to store rank distances
    total_chosen_acc = 0.0
    total_best_acc = 0.0
    
    # Add top-k accuracy tracking
    total_topk_correct = {k: 0 for k in topk_values}

    with torch.no_grad():
        for clips_batch, no_df_dice_batch, post_df_dice_batch, video_name_batch in loader:
            clips_batch = clips_batch.to(device)                        # [B, T, C, H, W]
            no_df_dice_batch = no_df_dice_batch.to(device)              # [B]
            post_df_dice_batch = post_df_dice_batch.to(device)          # [B, n_e]

            # Predict deferral logits
            #input= clips_batch.permute(0, 2, 1, 3, 4)
            rej_logits = model(clips_batch)

            # Calculate validation loss using deferral_loss
            val_loss = mao_regression_mae_loss(no_df_dice_batch, rej_logits, post_df_dice_batch, beta, distance_loss)
            total_val_loss += val_loss.item()

            # Inference based on rule: defer or not
            chosen_actions = infer_deferral_action(rej_logits)          # [B], 0 = no def, 1... = defer to frame j-1

            # Calculate accuracy metrics
            chosen_actions = infer_deferral_action(rej_logits)
            all_accs = torch.cat([no_df_dice_batch.unsqueeze(1), post_df_dice_batch], dim=1)
            # Accuracy from chosen action
            chosen_accs = torch.gather(all_accs, 1, chosen_actions.unsqueeze(1)).squeeze(1)


            # Calculate adjusted gain by subtracting beta from post_df_dice_batch
            adjusted_gain = post_df_dice_batch - beta - distance_loss
            # All possible accuracies: base + n_e frames with adjusted gain
            all_accs_adjusted = torch.cat([no_df_dice_batch.unsqueeze(1), adjusted_gain], dim=1)
            # Best accuracy (oracle) using argmax on adjusted gains
            best_actions = torch.argmax(all_accs_adjusted, dim=1)
            best_accs = torch.gather(all_accs, 1, best_actions.unsqueeze(1)).squeeze(1)

            # Compute metrics
            correct += (chosen_actions == best_actions).sum().item()
            regret = torch.abs(best_accs - chosen_accs)
            total_regret += regret.sum().item()
            total_samples += clips_batch.size(0)
            total_chosen_acc += chosen_accs.sum().item()
            total_best_acc += best_accs.sum().item()
            
            # Compute top-k accuracy
            topk_accuracies = calculate_topk_accuracy(rej_logits, best_actions, topk_values)
            for k, acc in topk_accuracies.items():
                total_topk_correct[k] += acc * clips_batch.size(0)
            
            
             # Compute rank distance per sample in batch
            for i in range(all_accs.size(0)):
                accs = all_accs[i]
                chosen_idx = chosen_actions[i].item()
                sorted_indices = torch.argsort(accs, descending=True)
                rank = (sorted_indices == chosen_idx).nonzero(as_tuple=True)[0].item()
                rank_distances.append(rank)

            # Store results
            all_best_actions.append(best_actions.cpu())
            all_chosen_actions.append(chosen_actions.cpu())
            all_video_names.extend(video_name_batch)  # Collect video names

    selection_accuracy = correct / total_samples
    mean_regret = total_regret / total_samples
    avg_val_loss = total_val_loss / len(loader)
    avg_rank_distance = sum(rank_distances) / len(rank_distances)
    all_best_actions = torch.cat(all_best_actions)
    all_chosen_actions = torch.cat(all_chosen_actions)
    avg_chosen_acc = total_chosen_acc / total_samples
    avg_best_acc = total_best_acc / total_samples
    
    # Calculate top-k accuracies
    topk_accuracies = {k: total_topk_correct[k] / total_samples for k in topk_values}

    return avg_val_loss, selection_accuracy, mean_regret, all_best_actions, all_chosen_actions, avg_rank_distance, avg_chosen_acc, avg_best_acc, topk_accuracies, all_video_names


# def validate_one_epoch(model, loader, criterion, device,logging):
#     model.eval()  # Set model to evaluation mode
#     correct = 0
#     total_regret = 0.0
#     total_samples = 0

#     with torch.no_grad():
#         for clips_batch, no_df_dice_batch, post_df_dice_batch, video_name_batch in loader:
#             clips_batch = clips_batch.to(device)
#             no_df_dice_batch = no_df_dice_batch.to(device)
#             post_df_dice_batch = post_df_dice_batch.to(device)

#             rej_logits = model(clips_batch.permute(0, 2, 1, 3, 4))
         
#             chosen_actions = torch.argmax(rej_logits, dim=1)
            
#             # Get best actions
#             all_accs = torch.cat([no_df_dice_batch.unsqueeze(1), post_df_dice_batch], dim=1)
#             best_actions = torch.argmax(all_accs, dim=1)
#             # logging.info("Best Actions:", best_actions)
#             # logging.info("Chosen Actions:", chosen_actions)
            
#             # Calculate metrics
#             correct += (chosen_actions == best_actions).sum().item()
            
#             # Calculate regret
#             chosen_accs = torch.gather(all_accs, 1, chosen_actions.unsqueeze(1)).squeeze(1)
#             best_accs = torch.gather(all_accs, 1, best_actions.unsqueeze(1)).squeeze(1)
#             regret = best_accs - chosen_accs
#             total_regret += regret.sum().item()
            
#             total_samples += clips_batch.shape[0]
    
#     selection_accuracy = correct / total_samples
#     mean_regret = total_regret / total_samples
    
#     return selection_accuracy, mean_regret, best_actions, chosen_actions

            


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    input_frame_inds,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
    save_palette_png=False,
):
    """Run inference on a single video with the given predictor."""
    # load the video frames and initialize the inference state on this video
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names = list(sorted(frame_names))
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    predictor.reset_state(inference_state)
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None
    
        
    # check and make sure we got at least one input frame
    if len(input_frame_inds) == 0:
        raise RuntimeError(
            f"In {video_name=}, got no input masks in {input_mask_dir=}. "
            "Please make sure the input masks are available in the correct format."
        )
    input_frame_inds = sorted(set(input_frame_inds))

    # add those input masks to SAM 2 inference state before propagation
    
    object_ids_set = None
    for input_frame_idx in input_frame_inds:
        try:
            per_obj_input_mask, input_palette = load_masks_from_dir(
                input_mask_dir=input_mask_dir,
                video_name=video_name,
                frame_name=frame_names[input_frame_idx],
                per_obj_png_file=per_obj_png_file,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"In {video_name=}, failed to load input mask for frame {input_frame_idx=}. "
                "Please add the `--track_object_appearing_later_in_video` flag "
                "for VOS datasets that don't have all objects to track appearing "
                "in the first frame (such as LVOS or YouTube-VOS)."
            ) from e
        
        # get the list of object ids to track from the first input frame
        if object_ids_set is None:
            object_ids_set = set(per_obj_input_mask)
        for object_id, object_mask in per_obj_input_mask.items():
            # check and make sure no new object ids appear only in later frames
            if object_id not in object_ids_set:
                raise RuntimeError(
                    f"In {video_name=}, got a new {object_id=} appearing only in a "
                    f"later {input_frame_idx=} (but not appearing in the first frame). "
                    "Please add the `--track_object_appearing_later_in_video` flag "
                    "for VOS datasets that don't have all objects to track appearing "
                    "in the first frame (such as LVOS or YouTube-VOS)."
                )
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=object_mask,
            )
            
            #------------------Save Images------------------------------
            
            # os.makedirs(output_mask_dir, exist_ok=True)
            # plt.figure(figsize=(9, 6))
            # plt.title(f"frame {input_frame_idx}")
            # plt.imshow(Image.open(os.path.join(base_video_dir, video_name, f"{frame_names[input_frame_idx]}.jpg")))
            # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
            # print(out_mask_logits.shape)
            
            # # Save the visualization image
            # vis_path = os.path.join(output_mask_dir, f"vis_add_{frame_names[input_frame_idx]}.png")
            # plt.savefig(vis_path)
            # plt.close()  # Close the figure to free memory
            
            #-----------------Save Images - End-------------------------

        # check and make sure we have at least one object to track
        if object_ids_set is None or len(object_ids_set) == 0:
            raise RuntimeError(
                f"In {video_name=}, got no object ids on {input_frame_inds=}. "
                "Please add the `--track_object_appearing_later_in_video` flag "
                "for VOS datasets that don't have all objects to track appearing "
                "in the first frame (such as LVOS or YouTube-VOS)."
            )
        
        # run propagation throughout the video and collect the results in a dict
        output_palette = input_palette or DAVIS_PALETTE
        video_segments = {}  # video_segments contains the per-frame segmentation results
        confidence_scores = {}
        video_segments_logits = {}

        for out_frame_idx, out_obj_ids, out_mask_logits, object_score_logits in predictor.propagate_in_video(
            inference_state
        ):
            #print (out_frame_idx)
            per_obj_output_mask = {
                out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
            per_obj_output_mask_logits = {
                out_obj_id: (out_mask_logits[i]).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
            video_segments_logits[out_frame_idx] = per_obj_output_mask_logits
            video_segments[out_frame_idx] = per_obj_output_mask
            confidence_scores[out_frame_idx] = object_score_logits.to(torch.float32).cpu().numpy()
          
        #---------------------------------Save Prediction--------------------------------------  
        # vis_frame_stride = 1   
        # for out_frame_idx in range(input_frame_inds[0], len(frame_names), vis_frame_stride):
        #     frame_name = frame_names[out_frame_idx]
        #     # print(frame_name)
        #     # print(out_frame_idx)
        #     # Load RGB frame
        #     img = Image.open(os.path.join(base_video_dir, video_name, f"{frame_name}.jpg"))

        #     # Load ground truth mask image (you can convert it to grayscale if needed)
        #     gt_mask_path = os.path.join(input_mask_dir, video_name,f"{frame_name}.png")
        #     gt_mask = Image.open(gt_mask_path).convert("L")  # grayscale mask

        #     fig, ax = plt.subplots(figsize=(8, 6))
        #     #fig.suptitle(f"Frame {out_frame_idx}", fontsize=14)

        #     # Show the input image
        #     ax.imshow(img)
        #     ax.set_title("Predicted + Ground Truth")
        #     ax.axis("off")  

        #     # Convert ground truth to NumPy and normalize to [0,1]
        #     gt_mask_np = np.array(gt_mask) / 255.0

        #     # Create transparent green overlay
        #     green_overlay = np.zeros((gt_mask_np.shape[0], gt_mask_np.shape[1], 4))
        #     green_overlay[..., 1] = 1.0  # green channel
        #     green_overlay[..., 3] = gt_mask_np * 0.4  # alpha based on mask

        #     # Overlay ground truth
        #     ax.imshow(green_overlay)

        #     # Show predicted masks
        #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #         show_mask(out_mask, ax, obj_id=out_obj_id)

        #     save_path = os.path.join(output_mask_dir, f"{frame_name}_vis.png")
        #     plt.tight_layout()
        #     plt.savefig(save_path, dpi=150)
            
        #     plt.close(fig) 
         #---------------------------------Save Prediction - END --------------------------------------  
        
    predictor.reset_state(inference_state)

    # # write the output masks as palette PNG files to output_mask_dir
    # for out_frame_idx, per_obj_output_mask in video_segments.items():
    #     if save_palette_png:
    #         # save palette PNG prediction results
    #         save_palette_masks_to_dir(
    #             output_mask_dir=output_mask_dir,
    #             video_name=video_name,
    #             frame_name=frame_names[out_frame_idx],
    #             per_obj_output_mask=per_obj_output_mask,
    #             height=height,
    #             width=width,
    #             per_obj_png_file=per_obj_png_file,
    #             output_palette=output_palette,
    #             confidence_scores=confidence_scores[out_frame_idx][0],
    #         )
    #     else:
    #         # save raw prediction results
    #         save_masks_to_dir(
    #             output_mask_dir=output_mask_dir,
    #             video_name=video_name,
    #             frame_name=frame_names[out_frame_idx],
    #             per_obj_output_mask=per_obj_output_mask,
    #             height=height,
    #             width=width,
    #             per_obj_png_file=per_obj_png_file,
    #             confidence_scores=confidence_scores[out_frame_idx][0],
    #         )
        
    #     print(f"confidence_scores frame {frame_names[out_frame_idx]}: ", confidence_scores[out_frame_idx][0])
    
    return video_segments_logits, confidence_scores

        
def calculate_auc(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predicted_probs = []

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels, delta_l_batch in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get model predictions
            inputs = inputs.repeat(1, 3, 1, 1, 1)
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs).cpu().numpy()  # Assuming binary classification

            true_labels.extend(labels.cpu().numpy())
            predicted_probs.extend(probabilities)

    # Calculate AUC
    auc = roc_auc_score(true_labels, predicted_probs)
    return auc

def plot_roc_curve(model, data_loader, device, output_dir):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predicted_probs = []

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels, delta_l_batch in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get model predictions
            inputs = inputs.repeat(1, 3, 1, 1, 1)
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs).cpu().numpy()  # Assuming binary classification

            true_labels.extend(labels.cpu().numpy())
            predicted_probs.extend(probabilities)

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the plot to the specified directory
    roc_curve_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_curve_path)
    plt.close()  # Close the figure to free memory

def calculate_confusion_matrix(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predicted_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels, delta_l_batch in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get model predictions
            inputs = inputs.repeat(1, 3, 1, 1, 1)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()  
            # Assuming binary classification
            preds_int = [int(pred[0]) for pred in preds]

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds_int)

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    true_labels_count = Counter(true_labels)
    predicted_labels_count = Counter(predicted_labels)
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    return true_labels_count, predicted_labels_count, cm_df

def plot_and_save_loss_accuracy_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    # Create a figure for the plots
    plt.figure(figsize=(10, 5))
    
    # Plot Loss Curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy Curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Save the plot to the specified directory
    plot_path = os.path.join(output_dir, 'loss_accuracy_curves.png')
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory

def get_last_commit_hash():
    """Get the last commit hash from git."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "Unknown (not a git repository or git command failed)"

def get_current_branch():
    """Get the current git branch name."""
    try:
        result = subprocess.run(['git', 'branch', '--show-current'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "Unknown (not a git repository or git command failed)"

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

def init_weights(m):
    """
    Initialize weights for the model using Kaiming initialization.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    

def log_memory_usage(device, epoch=None, logger=None, writer=None):
    """
    Logs GPU and CPU memory usage.

    Args:
        device: torch.device (usually 'cuda' or 'cuda:0')
        epoch: current epoch (for logging purposes)
        logger: optional logging module
        writer: optional TensorBoard writer
    """
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        gpu_max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        gpu_max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        gpu_memory_allocated = gpu_memory_reserved = gpu_max_allocated = gpu_max_reserved = 0.0

    cpu_memory_used = psutil.Process().memory_info().rss / (1024 ** 2)

    msg = (
        f"Memory Usage [Epoch {epoch+1 if epoch is not None else '-'}]: "
        f"GPU Allocated: {gpu_memory_allocated:.2f}MB | "
        f"GPU Reserved: {gpu_memory_reserved:.2f}MB | "
        f"GPU Max Allocated: {gpu_max_allocated:.2f}MB | "
        f"GPU Max Reserved: {gpu_max_reserved:.2f}MB | "
        f"CPU Used: {cpu_memory_used:.2f}MB"
    )

    print(msg)
    if logger:
        logger.info(msg)

    if writer and epoch is not None:
        writer.add_scalar('Memory/GPU_Allocated_MB', gpu_memory_allocated, epoch)
        writer.add_scalar('Memory/GPU_Reserved_MB', gpu_memory_reserved, epoch)
        writer.add_scalar('Memory/GPU_MaxAllocated_MB', gpu_max_allocated, epoch)
        writer.add_scalar('Memory/GPU_MaxReserved_MB', gpu_max_reserved, epoch)
        writer.add_scalar('Memory/CPU_MB', cpu_memory_used, epoch)

    # Reset peak stats for next epoch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        
def build_r2plus1d_model(num_classes=4, dropout_p=0.5, freeze_until='layer3'):
    # Load pretrained R(2+1)D model
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)

    # Modify first convolutional layer to accept 1-channel input
    old_conv = model.stem[0]
    new_conv = nn.Conv3d(
        in_channels=2,  # change if using different input channels
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )

    # Copy pretrained weights
    with torch.no_grad():
        rgb_weights = old_conv.weight  # [out_channels, 3, T, H, W]
    
        # Initialize both grayscale channels with the average of RGB pretrained weights
        gray_avg = rgb_weights.mean(dim=1, keepdim=True)  # shape: [out, 1, T, H, W]

        # Broadcast average weights to both grayscale channels
        new_conv.weight[:, 0, :, :, :] = gray_avg[:, 0, :, :, :]
        new_conv.weight[:, 1, :, :, :] = gray_avg[:, 0, :, :, :]

        # 3. Copy bias if exists
        if old_conv.bias is not None:
            new_conv.bias[:] = old_conv.bias

    # Replace the first conv layer
    model.stem[0] = new_conv

    # Freeze early layers
    freeze_layers = ['stem', 'layer1', 'layer2', 'layer3']
    for name, module in model.named_children():
        if name in freeze_layers:
            for param in module.parameters():
                param.requires_grad = False

    # Replace final fully connected layer with Dropout + Linear
   # Modify final FC layer
    if dropout_p is not None and dropout_p > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(model.fc.in_features, num_classes)     
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, train_accs, val_accs, 
                    current_ma_val_acc, best_ma_val_acc, args, timestamp, save_dir, experiment_name):
    """
    Save model checkpoint and training history.
    
    Args:
        model: The model to save
        optimizer: The optimizer state to save
        epoch: Current epoch number
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        current_ma_val_acc: Current moving average validation accuracy
        args: Training arguments
        timestamp: Timestamp for the run
        save_dir: Directory to save the checkpoint
        experiment_name: Name of the experiment
    """
    # Save model checkpoint with all necessary information
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    # Delete existing checkpoint file with the same experiment name
    old_checkpoint = os.path.join(save_dir, f"model_{experiment_name}_last_epoch_*.pth")
    try:
        existing_files = glob.glob(old_checkpoint)
        if existing_files:
            os.remove(existing_files[0])
            logging.info(f"Deleted old checkpoint: {existing_files[0]}")
            print(f"Deleted old checkpoint: {existing_files[0]}")
    except Exception as e:
        logging.warning(f"Failed to delete old checkpoint: {str(e)}")
        print(f"Failed to delete old checkpoint: {str(e)}")
    
    
    # Save new checkpoint
    checkpoint_path = os.path.join(save_dir, f"model_{experiment_name}_last_epoch_{epoch}_ma_acc_{current_ma_val_acc:.4f}.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'val_acc_ma': current_ma_val_acc,
        'best_epoch': epoch,
        'args': vars(args),
        'timestamp': timestamp
    }
     # Delete existing history file with the same experiment name
    old_history = os.path.join(save_dir, f"history_{experiment_name}_last_epoch_*.json")
    try:
        existing_files = glob.glob(old_history)
        if existing_files:
            os.remove(existing_files[0])
            logging.info(f"Deleted old history: {existing_files[0]}")
            print(f"Deleted old history: {existing_files[0]}")
    except Exception as e:
        logging.warning(f"Failed to delete old history: {str(e)}")
        print(f"Failed to delete old history: {str(e)}")
        
    history_path = os.path.join(save_dir, f"history_{experiment_name}_last_epoch_{epoch}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    logging.info(f"Saved last epoch model epoch {epoch} with moving average validation accuracy: {current_ma_val_acc:.4f}")
    print(f"Saved last epoch model epoch {epoch} with moving average validation accuracy: {current_ma_val_acc:.4f}")
    
    if current_ma_val_acc >= best_ma_val_acc:
        logging.info(f"Saved new best model epoch {epoch} with moving average validation accuracy: {current_ma_val_acc:.4f}")
        print(f"Saved new best model epoch {epoch} with moving average validation accuracy: {current_ma_val_acc:.4f}")
        
        # Delete existing checkpoint file with the same experiment name
        old_checkpoint = os.path.join(save_dir, f"model_{experiment_name}_best_epoch_*.pth")
        try:
            existing_files = glob.glob(old_checkpoint)
            if existing_files:
                os.remove(existing_files[0])
                logging.info(f"Deleted old checkpoint: {existing_files[0]}")
                print(f"Deleted old checkpoint: {existing_files[0]}")
        except Exception as e:
            logging.warning(f"Failed to delete old checkpoint: {str(e)}")
            print(f"Failed to delete old checkpoint: {str(e)}")
            
        # Save new checkpoint
        checkpoint_path = os.path.join(save_dir, f"model_{experiment_name}_best_epoch_{epoch}_ma_acc_{current_ma_val_acc:.4f}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Delete existing history file with the same experiment name
        old_history = os.path.join(save_dir, f"history_{experiment_name}_best_epoch_*.json")
        try:
            existing_files = glob.glob(old_history)
            if existing_files:
                os.remove(existing_files[0])
                logging.info(f"Deleted old history: {existing_files[0]}")
                print(f"Deleted old history: {existing_files[0]}")
        except Exception as e:
            logging.warning(f"Failed to delete old history: {str(e)}")
            print(f"Failed to delete old history: {str(e)}")
            
        history_path = os.path.join(save_dir, f"history_{experiment_name}_best_epoch_{epoch}.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        
        
def main():
    # Start total runtime tracking
    total_start_time = time.time()
    parser = argparse.ArgumentParser()
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
        required=True,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--data_npz_dir_train",
        type=str,
        required=True,
        help="directory with npz for train",
    )   
    parser.add_argument(
        "--data_npz_dir_test",
        type=str,
        default=None,
        help="directory with npz for test",
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
        required=True,
        help="Name of the experiment for logging and identification purposes",
    )
    # Add new arguments for training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--full_run",
        type=bool,
        default=False,
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
        default=0.5,
        help="Dropout rate for the model (default: 0.5)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
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
        default=10,
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
        nargs='+',
        default=[1, 3, 5],
        help="Top-k values to track for accuracy (default: [1, 3, 5])",
    )
    
    
    
    args = parser.parse_args()
    
    is_cuda_available = check_cuda()

    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Add timestamp to the output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    tensor_dir = os.path.join(args.output_mask_dir, 'tensorboard_logs')
    os.makedirs(tensor_dir, exist_ok=True)
    
    # Create output directory with timestamp and experiment name
    args.output_mask_dir = os.path.join(args.output_mask_dir, f"{timestamp}_{args.experiment_name}")
    os.makedirs(args.output_mask_dir, exist_ok=True)
    
    # Set up logging configuration
    log_file = os.path.join(args.output_mask_dir, 'output.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    
    # Log the start of the experiment
    logging.info("="*50)
    logging.info(f"Starting experiment: {args.experiment_name}")
    logging.info(f"Timestamp: {timestamp}")
    logging.info("="*50)
    
    # Initialize TensorBoard writer if enabled
    writer = None
    if args.tensorboard_status:  
        os.makedirs(tensor_dir, exist_ok=True)
        run_name = f"{timestamp}_{args.experiment_name}"
        writer = SummaryWriter(log_dir=os.path.join(tensor_dir, run_name))
        logging.info(f"TensorBoard logs will be saved to: {tensor_dir}/{run_name}")
        print(f"TensorBoard logs will be saved to: {tensor_dir}/{run_name}")
        
    if args.wandb_status:
        wandb.init(
            # set the wandb project where this run will be logged
            project="L2D-Video",
            name= f"{timestamp}_{args.experiment_name}",
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.learning_rate,
                "architecture": args.experiment_name,
                "epochs": args.num_epochs,
                "batch_size": args.batch_size
            }
        )


    # Log the git commit hash and branch
    commit_hash = get_last_commit_hash()
    branch_name = get_current_branch()
    logging.info(f"Git branch: {branch_name}")
    logging.info(f"Git commit hash: {commit_hash}")
    print(f"Git branch: {branch_name}")
    print(f"Git commit hash: {commit_hash}")

    # Print all arguments
    logging.info("\nArguments:")
    print("\nArguments:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
        print(f"{arg}: {getattr(args, arg)}")
    logging.info("\n")
    print("\n")
    
    # Here we are not considering this scenario : vos_separate_inference_per_object
    assert not args.track_object_appearing_later_in_video
    
   
    # ----- Prepare R(2+1)D model -----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained R(2+1)D model
    model = build_r2plus1d_model(num_classes=args.num_classes, dropout_p=args.dropout)
    start_epoch = 0
    if args.load_model_path is not None:
        checkpoint = torch.load(args.load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        logging.info(f"Loaded model from {args.load_model_path} at epoch {start_epoch-1}")
        print(f"Loaded model from {args.load_model_path} at epoch {start_epoch-1}")
    
    model = model.to(device)

    train_loader, val_loader = get_dataloaders(args, batch_size=args.batch_size)

    # For multi-class classification, we use CrossEntropyLoss instead of BCEWithLogitsLoss
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    
    # Load optimizer state if loading from checkpoint
    if args.load_model_path is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
     
    #--------------------------Train Model----------------------------------
    
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    # Initialize moving average queue for validation accuracy
    val_acc_ma_queue = deque(maxlen=2)
    best_ma_val_acc = 0.0
    best_chosen_val_acc = 0.0
    best_epoch = 0
    distance_loss = []
    for t in range(1, 10):  # Adjust based on the length of the video 
        distace_cost = 0.3*(np.exp(-0.3 * (t - 1)))  #find good values for distance factor and value inside exp term
        distance_loss.append(distace_cost)
    distance_loss = torch.tensor(distance_loss, dtype=torch.float32)
    distance_loss = distance_loss.to(device)
    logging.info(f"Distance loss: {distance_loss}")
    
    for epoch in range(start_epoch, args.num_epochs):
        # Start epoch runtime tracking
        epoch_start_time = time.time()

        
        train_loss, train_acc, train_regret, train_best_actions, train_chosen_actions, train_avg_rank_distance, train_chosen_acc, train_best_acc, topk_accuracies, video_names = train_one_epoch(model,epoch, train_loader, criterion, optimizer, args.save_every, args.alpha, args.beta, device, args.topk_values, distance_loss)
        
        # Calculate epoch runtime
        epoch_runtime = time.time() - epoch_start_time
        
        if (epoch+1) % args.save_every == 0:
            
            val_loss, val_acc, mean_regret, val_best_actions, val_chosen_actions, val_avg_rank_distance, val_chosen_acc, val_best_acc, val_topk_accuracies, val_video_names = validate_one_epoch(model,epoch, val_loader, criterion, args.alpha, args.beta, device, logging, args.topk_values, distance_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Update moving average of validation accuracy
            val_acc_ma_queue.append(val_acc)
            current_ma_val_acc = sum(val_acc_ma_queue) / len(val_acc_ma_queue)
            
            if args.tensorboard_status:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
                writer.add_scalar('Accuracy/val_moving_avg', current_ma_val_acc, epoch)
                writer.add_scalar('Regret/train', train_regret, epoch)
                writer.add_scalar('Regret/val', mean_regret, epoch)
                writer.add_scalar('Time/epoch_runtime', epoch_runtime, epoch)
                writer.add_scalar('Rank Distance/train', train_avg_rank_distance, epoch)
                writer.add_scalar('Rank Distance/val', val_avg_rank_distance, epoch)
                writer.add_scalar('Accuracy/chosen_train', train_chosen_acc, epoch)
                writer.add_scalar('Accuracy/best_train', train_best_acc, epoch)
                writer.add_scalar('Accuracy/chosen_val', val_chosen_acc, epoch)
                writer.add_scalar('Accuracy/best_val', val_best_acc, epoch)
                # Add top-k accuracy tracking
                for k in args.topk_values:
                    writer.add_scalar(f'Top{k} Accuracy/train', topk_accuracies[k], epoch)
                    writer.add_scalar(f'Top{k} Accuracy/val', val_topk_accuracies[k], epoch)
               
            if args.wandb_status:
                wandb.log({
                    "Epoch": epoch+1,
                    "Train Loss": train_loss,
                    "Train Acc": train_acc,
                    "Val Loss": val_loss,
                    "Val Acc": val_acc,
                    "Train Regret": train_regret,
                    "Val Regret": mean_regret,
                    "Train Avg Rank Distance": train_avg_rank_distance,
                    "Val Avg Rank Distance": val_avg_rank_distance,
                    "Train Best Acc": train_best_acc,
                    "Train Chosen Acc": train_chosen_acc,
                    "Val Best Acc": val_best_acc,
                    "Val Chosen Acc": val_chosen_acc,
                })
                
                # Add top-k accuracies to wandb
                for k in args.topk_values:
                    wandb.log({f"Train Top{k} Accuracy": topk_accuracies[k]})
                    wandb.log({f"Val Top{k} Accuracy": val_topk_accuracies[k]})

            logging.info(f"Epoch [{epoch+1}/{args.num_epochs}] Train Loss: {train_loss:.6f} Train Acc: {train_acc:.4f} Val Loss: {val_loss:.6f} Val Acc: {val_acc:.4f} Train Regret: {train_regret:.4f} Val Regret: {mean_regret:.4f}")
            # # Log top-k accuracies
            train_topk_str = "/".join([f"{topk_accuracies[k]:.4f}" for k in args.topk_values])
            val_topk_str = "/".join([f"{val_topk_accuracies[k]:.4f}" for k in args.topk_values])
            logging.info(f"Epoch [{epoch+1}/{args.num_epochs}] Top{args.topk_values} Train: {train_topk_str} Val: {val_topk_str}")
            logging.info(f"Epoch [{epoch+1}/{args.num_epochs}] Runtime: {epoch_runtime:.2f} seconds")
            logging.info(f"Current Moving Average Val Acc (10 epochs): {current_ma_val_acc:.4f}")
            
            # Log training best action and chosen action for 10 samples with video names
            logging.info(f"Training Best Actions: {train_best_actions[:80]}")
            logging.info(f"Training Chosen Actions: {train_chosen_actions[:80]}")
            logging.info(f"Training Video Names: {video_names[:80]}")
            
          

            # Log validation best action and chosen action for 10 samples with video names
            logging.info(f"Validation Best Actions: {val_best_actions[:80]}")
            logging.info(f"Validation Chosen Actions: {val_chosen_actions[:80]}")
            logging.info(f"Validation Video Names: {val_video_names[:80]}")
            
            
            
            print(f"Epoch [{epoch+1}/{args.num_epochs}] Train Loss: {train_loss:.6f} Train Acc: {train_acc:.4f} Val Loss: {val_loss:.6f} Val Acc: {val_acc:.4f} Train Regret: {train_regret:.4f} Val Regret: {mean_regret:.4f}")
            # print(f"Epoch [{epoch+1}/{args.num_epochs}] Top{args.topk_values} Train: {train_topk_str} Val: {val_topk_str}")
            print(f"Epoch [{epoch+1}/{args.num_epochs}] Runtime: {epoch_runtime:.2f} seconds")
            print(f"Current Moving Average Val Acc (10 epochs): {current_ma_val_acc:.4f}")
            
            # Log memory usage
            log_memory_usage(device, epoch=epoch, logger=logging, writer=writer if args.tensorboard_status else None)

        
            
            if args.save_model:
                
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    train_accs=train_accs,
                    val_accs=val_accs,
                    current_ma_val_acc=val_chosen_acc,
                    best_ma_val_acc=best_chosen_val_acc,
                    args=args,
                    timestamp=timestamp,
                    save_dir=args.output_mask_dir,
                    experiment_name=args.experiment_name
                    )
                if val_chosen_acc > best_chosen_val_acc:
                    best_chosen_val_acc = val_chosen_acc

    # Calculate and log total runtime
    total_runtime = time.time() - total_start_time
    logging.info(f"Total training runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    print(f"Total training runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    # logging.info(f"Best model was from epoch {best_epoch} with moving average validation accuracy: {best_ma_val_acc:.4f}")
    # print(f"Best model was from epoch {best_epoch} with moving average validation accuracy: {best_ma_val_acc:.4f}")

    # print(f"completed inference on {len(video_names)} videos -- output masks saved to {args.output_mask_dir}")
    # logging.info(f"completed inference on {len(video_names)} videos -- output masks saved to {args.output_mask_dir}")
    

    # Plot and save loss and accuracy curves
    plot_and_save_loss_accuracy_curves(train_losses, val_losses, train_accs, val_accs, args.output_mask_dir)

    # Close TensorBoard writer if it was initialized
    if writer is not None:
        writer.close()
        logging.info("TensorBoard writer closed")

    logging.info("="*50)
    logging.info(f"Experiment completed: {args.experiment_name}")
    logging.info(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    logging.info("="*50)


if __name__ == "__main__":
    main()