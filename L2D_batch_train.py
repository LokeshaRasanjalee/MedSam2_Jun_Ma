# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from collections import defaultdict
import datetime
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import csv
import logging

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
import torchvision.transforms as T
import torch.optim as optim
import torchvision.models.video as models
from PIL import Image
from scipy.ndimage import label, center_of_mass
from matplotlib.patches import Rectangle

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

def get_mask_img_list(args, frame_names, video_name):
    mask_img_list = [
        int(name)  # convert string like '0000' → integer 0
        for idx, name in enumerate(frame_names)
        if os.path.exists(
            os.path.join(args.input_mask_dir, video_name, f"{name}.png")
        )
    ]
    return mask_img_list


# def compute_focal_loss(pred_mask, true_mask, alpha=0.25, gamma=2.0, eps=1e-6):
#     if isinstance(pred_mask, np.ndarray):
#         pred_mask = torch.from_numpy(pred_mask)
#     if isinstance(true_mask, np.ndarray):
#         true_mask = torch.from_numpy(true_mask)

#     pred_mask = pred_mask.float()
#     true_mask = true_mask.float()

#     if true_mask.ndim == 2:
#         true_mask = true_mask.unsqueeze(0)  # match shape: [1, H, W]

#     prob = torch.sigmoid(pred_mask)
#     prob = prob.clamp(min=eps, max=1. - eps)

#     ce_loss = F.binary_cross_entropy_with_logits(pred_mask, true_mask, reduction='none')
#     p_t = prob * true_mask + (1 - prob) * (1 - true_mask)
#     alpha_t = alpha * true_mask + (1 - alpha) * (1 - true_mask)
#     focal_weight = (1 - p_t) ** gamma

#     loss = alpha_t * focal_weight * ce_loss
#     return loss.mean()


# def dice_score(pred_mask, true_mask, eps=1e-5):
#     #print ("dice_score")
#     pred = pred_mask.flatten()
#     true = true_mask.flatten()
#     intersection = (pred * true).sum()
#     return (2. * intersection) / (pred.sum() + true.sum() + eps)

def dice_loss_calc(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    # Convert inputs to PyTorch tensor if it's a NumPy array
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).float()
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets).float()
        
    if inputs.shape != targets.shape:
        # If targets is 1D and inputs is 2D, expand targets
        if targets.dim() == 1 and inputs.dim() == 2:
            targets = targets.unsqueeze(0)
        # If targets is 2D and inputs is 3D, expand targets
        elif targets.dim() == 2 and inputs.dim() == 3:
            targets = targets.unsqueeze(0)
        # If targets is 3D and inputs is 4D, expand targets
        elif targets.dim() == 3 and inputs.dim() == 4:
            targets = targets.unsqueeze(0)
        
    #inputs = inputs.sigmoid() # Remove sigmoid for dice loss since it's already applied in the model
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss_calc(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    # Convert inputs to PyTorch tensor if it's a NumPy array
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).float()
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets).float()
    
    # Ensure inputs and targets have the same shape
    if inputs.shape != targets.shape:
        # If targets is 2D and inputs is 3D, expand targets
        if targets.dim() == 2 and inputs.dim() == 3:
            targets = targets.unsqueeze(0)
        # If targets is 3D and inputs is 4D, expand targets
        elif targets.dim() == 3 and inputs.dim() == 4:
            targets = targets.unsqueeze(0)
        # If targets has extra dimension, squeeze it
        elif targets.dim() > inputs.dim():
            targets = targets.squeeze(0)
            
    # prob = inputs.sigmoid()
    # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # Use inputs directly since they're already between 0 and 1
 
    prob = inputs
    ce_loss = F.binary_cross_entropy(prob, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects


def compute_frame_features(curr_mask, prev_mask, logit, confidence_score):
    #print ("compute_frame_features")
    dice_loss = dice_loss_calc(curr_mask, prev_mask, 1, loss_on_multimask=False)
    conf_mean = logit.mean().item()
    conf_std = logit.std().item()
    area = curr_mask.sum().item()
    #edge_sharpness = compute_edge_sharpness(curr_mask)
    
    return [dice_loss, conf_mean, conf_std, area, confidence_score]

def compute_iou(mask1, mask2, eps=1e-5):
    #print ("compute_iou")
    intersection = ((mask1 > 0) & (mask2 > 0)).sum()
    union = ((mask1 > 0) | (mask2 > 0)).sum()
    #print ("Intersection: ",intersection,"     |     Union: ", union)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0  # Special case: both empty = perfect match
    else:
        return intersection / (union)


def train_deferral_model(X, y):
    print("train_deferral_model")
    logging.info("train_deferral_model")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = LogisticRegression()
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    y_pred_prob = clf.predict_proba(X_test_scaled)
    for i, prob in enumerate(y_pred_prob):
        logging.info(f"Test case {i+1}: Probability of positive class = {prob[1]:.3f}")
        print(f"Test case {i+1}: Probability of positive class = {prob[1]:.3f}")
    logging.info(f"Model accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Model accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
    return clf

def train_regression_model(X, y):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("Regression Model MSE: ", mean_squared_error(y_test, y_pred))
    return reg

def save_model(model, output_path, model_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    joblib.dump(model, os.path.join(output_path, model_name + ".pkl"))
 
    
def downstream_impact(fir_prom, sec_prom, pred_logits_uncorrected, pred_logits_corrected, gt_masks,score_thresh, K):
    #consider the entire impact on the video, not just specific to a region
    future_iou_u, future_iou_c = [], []
    T = len(pred_logits_uncorrected)

    end = min(sec_prom+K, fir_prom+T) # Correct here

    for k in range(sec_prom,end):
        mask_u = (pred_logits_uncorrected[k][1] > score_thresh).astype(float)
        mask_c = (pred_logits_corrected[k][1] > score_thresh).astype(float)
        gt = gt_masks[k]

        future_iou_u.append(compute_iou(mask_u, gt))
        future_iou_c.append(compute_iou(mask_c, gt))

    #impact = ((np.mean(future_iou_c) - np.mean(future_iou_u))/( np.mean(future_iou_u)+  1e-5))*100
    impact = (np.mean(future_iou_c) - np.mean(future_iou_u))
    #print (impact)
    return impact, np.mean(future_iou_c), np.mean(future_iou_u)


def compute_downstream_loss(video_segments, gt_list, frame_indices_for_clip):
    """
    Compute downstream loss for the given frame indices.
    
    video_segments: list of predicted masks (numpy arrays), either uncorrected or corrected
    gt_list: list of ground truth masks (numpy arrays)
    frame_indices_for_clip: list of frame indices to calculate IoU # PASS THE LIST OF INDICES
    """
    total_dice_loss = 0.0
    total_sam_loss = 0.0
    total_focal_loss = 0.0
    # dice_loss_list = []
    # sam_loss_list = []
    # focal_loss_list = []
    # valid_frames = 0
    
    if video_segments == None:
        return None,None,None    
    
    for idx in frame_indices_for_clip:
        # if idx >= len(video_segments) or idx >= len(gt_list):
        #     continue  # Skip out of range indices

        pred_mask = video_segments[idx]  # Assuming your video_segments store (frame_index, mask) tuples
        gt_mask = gt_list[idx][0]            # Your gt_list stores (1, H, W) numpy arrays

        
        dice_loss = dice_loss_calc(pred_mask, gt_mask, 1, loss_on_multimask=False)  # inputs, targets, num_objects, loss_on_multimask=False
        focal_loss = sigmoid_focal_loss_calc(pred_mask, gt_mask, 1, loss_on_multimask=False)
        sam_loss = dice_loss + 20*focal_loss

        total_dice_loss += dice_loss
        total_focal_loss += focal_loss
        total_sam_loss += sam_loss
        
        # dice_loss_list.append(dice_loss)
        # focal_loss_list.append(focal_loss)
        # sam_loss_list.append(sam_loss)
        # valid_frames += 1
        

    # assert valid_frames != 0

    avg_dice_loss = total_dice_loss / len(frame_indices_for_clip)
    avg_sam_loss = total_sam_loss / len(frame_indices_for_clip)
    avg_focal_loss = total_focal_loss / len(frame_indices_for_clip)
    #downstream_loss = 1.0 - avg_iou
    return avg_dice_loss, avg_sam_loss, avg_focal_loss


def add_mask(input_mask_dir,output_mask_dir,base_video_dir, video_name, frame_names, 
             input_frame_idx, object_ids_set, per_obj_png_file,predictor,inference_state):
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
        
        os.makedirs(output_mask_dir, exist_ok=True)
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {input_frame_idx}")
        plt.imshow(Image.open(os.path.join(base_video_dir, video_name, f"{frame_names[input_frame_idx]}.jpg")))
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
       
        
        # Save the visualization image
        vis_path = os.path.join(output_mask_dir, f"vis_add_mask_{frame_names[input_frame_idx]}.png")
        plt.savefig(vis_path)
        plt.close()  # Close the figure to free memory
        
        #-----------------Save Images - End-------------------------
    return  out_obj_ids, out_mask_logits, object_ids_set


def add_point(input_mask_dir,output_mask_dir,base_video_dir, video_name, frame_names, 
             input_frame_idx, object_ids_set, per_obj_png_file,predictor,inference_state,
             blob_centers, labels,gt):
        
    _, out_obj_ids, out_mask_logits =  predictor.add_new_points_or_box(
        inference_state = inference_state,
        frame_idx = input_frame_idx,
        obj_id = 1,
        points=blob_centers,
        labels=labels,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    )
    
    #------------------Save Images------------------------------
    
    # os.makedirs(output_mask_dir, exist_ok=True)
    # plt.figure(figsize=(9, 6))
    # plt.title(f"frame {input_frame_idx}")
    # plt.imshow(Image.open(os.path.join(base_video_dir, video_name, f"{frame_names[input_frame_idx]}.jpg")))
    
    # # Overlay the ground truth mask in green
    # gt_mask = gt[input_frame_idx]
    # gt_mask = np.squeeze(gt_mask)  # Ensure it's 2D
    # green_overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 4))
    # green_overlay[..., 1] = 1.0  # green channel
    # green_overlay[..., 3] = gt_mask * 0.4  # alpha based on mask
    # plt.imshow(green_overlay)
    
    # # Show the predicted mask on top
    # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    
    # # Mark the blob centers on the image with a cross
    # for center in blob_centers:
    #     plt.plot(center[0], center[1], 'rx')  # 'rx' for red crosses
    
    
    
    # # Save the visualization image
    # timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # if labels == 1:
    #     vis_path = os.path.join(output_mask_dir, f"vis_add_positive_{frame_names[input_frame_idx]}_{center[0]:.3f}_{center[1]:.3f}_{timestamp}.png")
    # elif labels == 0:
    #     vis_path = os.path.join(output_mask_dir, f"vis_add_negative_{frame_names[input_frame_idx]}_{center[0]:.3f}_{center[1]:.3f}_{timestamp}.png")
    # plt.savefig(vis_path)
    # plt.close()  # Close the figure to free memory
    
    #-----------------Save Images - End-------------------------
    return out_obj_ids,out_mask_logits                     

def add_box(input_mask_dir,output_mask_dir,base_video_dir, video_name, frame_names, 
             input_frame_idx, object_ids_set, per_obj_png_file,predictor,inference_state,
             bbox, labels,gt):
        
    _, out_obj_ids, out_mask_logits =  predictor.add_new_points_or_box(
        inference_state = inference_state,
        frame_idx = input_frame_idx,
        obj_id = 1,
        box=bbox,
    )
    
    #------------------Save Images------------------------------
    
    os.makedirs(output_mask_dir, exist_ok=True)
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {input_frame_idx}")
    plt.imshow(Image.open(os.path.join(base_video_dir, video_name, f"{frame_names[input_frame_idx]}.jpg")))
    
    # Overlay the ground truth mask in green
    gt_mask = gt[input_frame_idx]
    gt_mask = np.squeeze(gt_mask)  # Ensure it's 2D
    green_overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 4))
    green_overlay[..., 1] = 1.0  # green channel
    green_overlay[..., 3] = gt_mask * 0.4  # alpha based on mask
    plt.imshow(green_overlay)
    
    # Show the predicted mask on top
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    
    # Mark the blob centers on the image with a cross
    x_min, y_min, x_max, y_max = bbox[0]
    width = x_max - x_min
    height = y_max - y_min
    rect = Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    
    print(out_mask_logits.shape)
    
    # Save the visualization image
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if labels == 1:
        vis_path = os.path.join(output_mask_dir, f"vis_add_box_{frame_names[input_frame_idx]}_{x_min:.3f}_{y_min:.3f}_{x_max:.3f}_{y_max:.3f}_{timestamp}.png")
    plt.savefig(vis_path)
    plt.close()  # Close the figure to free memory
    
    #-----------------Save Images - End-------------------------
    return out_obj_ids,out_mask_logits  

def find_blob_centers(mask):
    """Find the center points of blobs in a binary mask."""
    # Label the blobs in the mask
    labeled_mask, num_features = label(mask)
    
    # Calculate the center of mass for each blob
    centers = center_of_mass(mask, labeled_mask, range(1, num_features + 1))
    
    return centers

def keep_largest_blob(mask):
    labeled, num = label(mask)
    if num == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    largest_label = sizes.argmax()
    return labeled == largest_label

# def get_bounding_box(mask):
#     """Return (x_min, y_min, x_max, y_max) of the foreground blob in a binary mask."""
#     ys, xs = np.where(mask)  # y = row, x = column
#     if len(xs) == 0 or len(ys) == 0:
#         return None  # no foreground
#     x_min, x_max = xs.min(), xs.max()
#     y_min, y_max = ys.min(), ys.max()
#     return (x_min, y_min, x_max, y_max)

def get_bounding_box(mask):
    """Return (x_min, y_min, x_max, y_max) of a box triple the size of the foreground blob in a binary mask."""
    ys, xs = np.where(mask)  # y = row, x = column
    if len(xs) == 0 or len(ys) == 0:
        return None  # no foreground

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Original box center, width, and height
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    # Triple the size
    new_width = width * 2
    new_height = height * 2

    # New box coordinates
    new_x_min = int(round(cx - new_width / 2))
    new_x_max = int(round(cx + new_width / 2)) - 1
    new_y_min = int(round(cy - new_height / 2))
    new_y_max = int(round(cy + new_height / 2)) - 1

    # Clip to image boundaries
    H, W = mask.shape
    new_x_min = max(0, new_x_min)
    new_y_min = max(0, new_y_min)
    new_x_max = min(W - 1, new_x_max)
    new_y_max = min(H - 1, new_y_max)

    return (new_x_min, new_y_min, new_x_max, new_y_max)


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    input_frame_inds,
    gt,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
    save_palette_png=False,
    prompt="mask",
    
):
    
    print ("input_frame_inds: ", input_frame_inds)
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
    # height = inference_state["video_height"]
    # width = inference_state["video_width"]
    input_palette = None
    
        
    # check and make sure we got at least one input frame
    if len(input_frame_inds) == 0:
        raise RuntimeError(
            f"In {video_name=}, got no input masks in {input_mask_dir=}. "
            "Please make sure the input masks are available in the correct format."
        )
    input_frame_inds = sorted(input_frame_inds)
    
    #Then 

    
    object_ids_set = None # Bipass the object id check
    
    out_mask_logits=[]
    out_frame_idx=[]
    for idx, input_frame_idx in enumerate(input_frame_inds):
        print ("input_frame_idx: ", input_frame_idx)
        if idx == 0:
            if prompt[0] == "mask":
                print(f"{input_frame_idx} -Add mask")
                out_obj_ids_prompt, out_mask_logits_prompt,object_ids_set = add_mask(input_mask_dir,output_mask_dir,base_video_dir, video_name, frame_names, 
                input_frame_idx, object_ids_set, per_obj_png_file,predictor,inference_state)
            
            elif prompt[0] == "point":
                print(f"{input_frame_idx} -Add point")
                
                cleaned_mask = keep_largest_blob(gt[input_frame_idx][0])
                labeled_cleaned, n_clean = label(cleaned_mask)
                if n_clean != 1:
                    raise SystemExit("Exiting with error due to multiple objects in the first frame.")
                    print("Multiple objects in the first frame.")
                    return None, None  
                center = center_of_mass(cleaned_mask)[::-1]
                labels = np.ones(1)
                
                #-----------Object ID Check -----------------
                #This is to bipass multiobject scenario. Change it when you work with multiple objects
                # try:
                #     per_obj_input_mask, input_palette = load_masks_from_dir(
                #         input_mask_dir=input_mask_dir,
                #         video_name=video_name,
                #         frame_name=frame_names[input_frame_idx],
                #         per_obj_png_file=per_obj_png_file,
                #     )
                # except FileNotFoundError as e:
                #     raise RuntimeError(
                #         f"In {video_name=}, failed to load input mask for frame {input_frame_idx=}. "
                #         "Please add the `--track_object_appearing_later_in_video` flag "
                #         "for VOS datasets that don't have all objects to track appearing "
                #         "in the first frame (such as LVOS or YouTube-VOS)."
                #     ) from e
                
                # # get the list of object ids to track from the first input frame
                # if object_ids_set is None:
                #     object_ids_set = set(per_obj_input_mask)
                    
                # if len(object_ids_set) != 1:
                #     raise SystemExit("Exiting with error due to multiple objects in the first frame.")
                #     print("Multiple objects in the first frame.")
                #     return None, None
                    
                #-----------Object ID Check - End -----------------
                    
                
                
                
                
                
                if center:
                    out_obj_ids_prompt,out_mask_logits_prompt = add_point(input_mask_dir, output_mask_dir, base_video_dir, video_name, frame_names, 
                    input_frame_idx, object_ids_set, per_obj_png_file, predictor, inference_state,
                              [center], labels,gt)
                    object_ids_set = [1]
                else:
                    raise SystemExit("Exiting with error due to no mask or negative points.")
                    print("No mask or negative points")
                    return None, None   
                
            elif prompt[0]  == "box":
                print(f"{input_frame_idx} -Add box")
                
                cleaned_mask = keep_largest_blob(gt[input_frame_idx][0])
                labeled_cleaned, n_clean = label(cleaned_mask)
                if n_clean != 1:
                    raise SystemExit("Exiting with error due to multiple objects in the first frame.")
                    print("Multiple objects in the first frame.")
                    return None, None  
                
                bbox = get_bounding_box(cleaned_mask)
                labels = np.ones(1)
                if bbox:
                    out_obj_ids_prompt,out_mask_logits_prompt = add_box(input_mask_dir, output_mask_dir, base_video_dir, video_name, frame_names, 
                    input_frame_idx, object_ids_set, per_obj_png_file, predictor, inference_state,
                              [bbox], labels,gt)
                    object_ids_set = [1]
                else:
                    raise SystemExit("Exiting with error due to no mask or negative points.")
                    print("No mask or negative points")
                    return None, None   
                
                
                
                    
                
        else:
            #Check if the output mask logit for relavant frame id have a gt mask or not
            
            # if gt has a mask
            if np.any(gt[input_frame_idx] == 1):
                
                if prompt[1] == "mask":
                
                    print(f"{input_frame_idx} - Add mask")
                    out_obj_ids_prompt, out_mask_logits_prompt,object_ids_set= add_mask(input_mask_dir,output_mask_dir,base_video_dir, video_name, frame_names, 
                    input_frame_idx, object_ids_set, per_obj_png_file,predictor,inference_state)
                    
                elif prompt[1] == "point":
                    print(f"{input_frame_idx} -Add point")
                    cleaned_mask = keep_largest_blob(gt[input_frame_idx][0])
                    labeled_cleaned, n_clean = label(cleaned_mask)
                    if n_clean != 1:
                        raise SystemExit("Exiting with error due to multiple objects in the first frame.")
                        print("Multiple objects in the first frame.")
                        return None, None  
                    center = center_of_mass(cleaned_mask)[::-1]
                    labels = np.ones(1)
                    
                    if center:
                        out_obj_ids_prompt,out_mask_logits_prompt = add_point(input_mask_dir, output_mask_dir, base_video_dir, video_name, frame_names, 
                        input_frame_idx, object_ids_set, per_obj_png_file, predictor, inference_state,
                                [center], labels,gt)
                        object_ids_set = [1]
                    else:
                        raise SystemExit("Exiting with error due to no mask or negative points.")
                        print("No mask or negative points")
                        return None, None  
                
                elif prompt[1] == "box":
                    print(f"{input_frame_idx} -Add box")
                    
                    cleaned_mask = keep_largest_blob(gt[input_frame_idx][0])
                    labeled_cleaned, n_clean = label(cleaned_mask)
                    if n_clean != 1:
                        raise SystemExit("Exiting with error due to multiple objects in the first frame.")
                        print("Multiple objects in the first frame.")
                        return None, None  
                    
                    bbox = get_bounding_box(cleaned_mask)
                    labels = np.ones(1)
                    if bbox:
                        out_obj_ids_prompt,out_mask_logits_prompt = add_box(input_mask_dir, output_mask_dir, base_video_dir, video_name, frame_names, 
                        input_frame_idx, object_ids_set, per_obj_png_file, predictor, inference_state,
                                [bbox], labels,gt)
                        object_ids_set = [1]
                    else:
                        raise SystemExit("Exiting with error due to no mask or negative points.")
                        print("No mask or negative points")
                        return None, None    
                    
                    
                
            else:
                
                
                if np.any(video_segments[input_frame_idx][1] == 1): 
                    print("Add negative points iteratively")
                    
                    blob_centers_list=[]
                    labels_list=[]
                    
                    pred_wrong_mask = np.squeeze(video_segments[input_frame_idx][1])
                    blob_centers = np.array(find_blob_centers(pred_wrong_mask))
                    blob_centers = blob_centers.astype(np.float32)[:, [1, 0]][0]
                    
                    while len(blob_centers) > 0:
                        
                        
                        blob_centers_list.append(list(blob_centers))
                        labels = np.zeros(len(blob_centers_list))
                        #labels_list.append(labels)
                        
                        # Remove the blob centers
                        out_mask_logits_prompt = add_point(
                            input_mask_dir, output_mask_dir, base_video_dir, video_name, frame_names, 
                            input_frame_idx, object_ids_set, per_obj_png_file, predictor, inference_state, 
                            blob_centers_list, labels,gt
                        )
                        obj_id = 1 #Bipass object id extraction
                        out_mask_logits_prompt = (out_mask_logits_prompt[obj_id] > score_thresh).cpu().numpy()
                        
                        # Recalculate blob centers after removal
                        pred_wrong_mask = np.squeeze(out_mask_logits_prompt[0, 0, :, :])
                        blob_centers = np.array(find_blob_centers(pred_wrong_mask))
                        
                        if len(blob_centers) != 0:
                            blob_centers = blob_centers.astype(np.float32)[:, [1, 0]][0]
                        else:
                            break
   
                    
                              
                else: 
                    # raise SystemExit("Exiting with error due to no mask or negative points.")
                    print("No mask or negative points")
                    return None, None
                    
                    
                
                
       
                
                
       

        #---------------------Mask Input End-----------------------------------
        #---------------------------------------------------------------------

        #---------------------Point Input Start--------------------------------


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
        vis_frame_stride = 1   
        for out_frame_idx in range(input_frame_inds[0], len(frame_names), vis_frame_stride):
            frame_name = frame_names[out_frame_idx]
            # print(frame_name)
            # print(out_frame_idx)
            # Load RGB frame
            img = Image.open(os.path.join(base_video_dir, video_name, f"{frame_name}.jpg"))

            # Load ground truth mask image (you can convert it to grayscale if needed)
            gt_mask_path = os.path.join(input_mask_dir, video_name,f"{frame_name}.png")
            gt_mask = Image.open(gt_mask_path).convert("L")  # grayscale mask

            fig, ax = plt.subplots(figsize=(8, 6))
            #fig.suptitle(f"Frame {out_frame_idx}", fontsize=14)

            # Show the input image
            ax.imshow(img)
            ax.set_title("Predicted + Ground Truth")
            ax.axis("off")  

            # Convert ground truth to NumPy and normalize to [0,1]
            gt_mask_np = np.array(gt_mask) / 255.0

            # Create transparent green overlay
            green_overlay = np.zeros((gt_mask_np.shape[0], gt_mask_np.shape[1], 4))
            green_overlay[..., 1] = 1.0  # green channel
            green_overlay[..., 3] = gt_mask_np * 0.4  # alpha based on mask

            # Overlay ground truth
            ax.imshow(green_overlay)

            # Show predicted masks
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, ax, obj_id=out_obj_id)

            save_path = os.path.join(output_mask_dir, f"{frame_name}_vis.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            
            plt.close(fig) 
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





def main():
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
        required=True,
        help="directory containing videos (as JPEG files) to run inference on",
    )
    parser.add_argument(
        "-m",
        "--input_mask_dir",
        type=str,
        required=True,
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
        "--post_hoc_model_save_dir",
        type=str,
        required=True,
        help="directory to save the post hoc model",
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
    parser.add_argument(
        "--array_id",
        type=int,
        required=True,
        help="array ID for the current batch processing",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help="number of chunks for the current batch processing",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs='+',
        default=["mask"],
        help="prompt type(s) for first prompt (e.g., mask point box). Pass one or more."
    )
    

    args = parser.parse_args()
    
    
    if torch.cuda.is_available():
        print("CUDA is available. Running on GPU:", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print("CUDA is NOT available. Running on CPU.")
        device = torch.device("cpu")
   
    
    # Add timestamp to the output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_mask_dir = os.path.join(args.output_mask_dir, f"{args.experiment_name}_{timestamp}")
    
    # Ensure the directory exists
    os.makedirs(args.output_mask_dir, exist_ok=True)

    
     # Set up logging to use the output_mask_dir
    logging.basicConfig(
        filename=os.path.join(args.output_mask_dir, 'output.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Print all arguments
    print("\nArguments:")
    logging.info("\nArguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
        logging.info(f"{arg}: {getattr(args, arg)}")
    print("\n")

    
    # if we use per-object PNG files, they could possibly overlap in inputs and outputs
    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false" if args.per_obj_png_file else "true")
    ]
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
        vos_optimized=args.use_vos_optimized_video_predictor,
    )

    if args.use_all_masks:
        print("using all available masks in input_mask_dir as input to the MedSAM2 model")
    else:
        print(
            "using only the first frame's mask in input_mask_dir as input to the MedSAM2 model"
        )
    # if a video list file is provided, read the video names from the file
    # (otherwise, we use all subdirectories in base_video_dir)
    if args.video_list_file is not None:
        with open(args.video_list_file, "r") as f:
            video_names = [v.strip() for v in f.readlines()]
    else:
        video_names = [
            p
            for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        ]
        
    # Here we are not considering this scenario : vos_separate_inference_per_object
    assert not args.track_object_appearing_later_in_video
    
    video_names.sort()
    # Calculate chunk size to get 2000 chunks
    num_groups = args.num_chunks
    total = len(video_names)

    # Compute approximate chunk size
    chunked = []
    chunk_size = total // num_groups
    remainder = total % num_groups

    start = 0
    for i in range(num_groups):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunked.append(video_names[start:end])
        start = end

    # Output number of elements in each chunk
    group_sizes = [len(group) for group in chunked]
    for i, size in enumerate(group_sizes):
        print(f"Group {i+1}: {size} items")

    # Output total
    print(f"\nTotal items across all groups: {sum(group_sizes)}")
    

    current_chunk = chunked[args.array_id] 
    
      # ----- Define Resize Transform for R(2+1)D -----
    timesformer_transform = T.Compose([
        T.ToTensor()            # Convert HWC -> CHW, float in [0, 1]
        
    ])
    
    
    
    #--------------------------Loop though vidoes----------------------------------


    window_size = 8  # number of frames per clip


    for n_video, video_name in enumerate(current_chunk):
        
        if video_name != '0005_0030':
            continue
               
       
        L_post_defer_list = []
        L_post_defer_sam_loss_list = []
        L_post_defer_focal_loss_list = []
        clips = []
        # ll_post_dice = []
        # ll_post_sam = []
        # ll_post_focal = []
        
        # if video_name != 'seq4':
        #     continue
        
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        logging.info(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        
        folder_name_list = []
        combined_scores = {}      # frame_index → {folder_name: confidence}
        frame_indices = set()
        
        frame_names = get_frame_names(os.path.join(args.base_video_dir, video_name))    
        mask_img_list_with_obj = sorted(get_mask_img_list(args, frame_names, video_name))
        frame_indices_for_clip = mask_img_list_with_obj.copy()
        initial_prompt = int(mask_img_list_with_obj[0])
        mask_img_list_with_obj.pop(0)
        
        
        #--------------------Get ground truth masks--------------------------------
        gt_list = []
        for mask_name in frame_names:
            input_mask_path = os.path.join(args.input_mask_dir, video_name, f"{mask_name}.png")
            if os.path.exists(input_mask_path):
                input_mask, _ = load_ann_png(input_mask_path)
                if input_mask.ndim == 3:
                # RGB mask → binary mask
                    gt = np.any(input_mask > 0, axis=-1)
                elif input_mask.ndim == 2:
                    # Already 2D, just ensure it's boolean
                    gt = input_mask > 0
                gt = np.expand_dims(gt, axis=0) 
                gt_list.append(gt)
            
        # -------------------- Prompt on First frame ------------------------------

        folder_name = str(initial_prompt)
        print ("Inital Prompt: ", folder_name)
        folder_name_list.append(folder_name)
        output_mask_dir = os.path.join(args.output_mask_dir, video_name, folder_name)
        
        video_segments_first, confidence_scores_first = vos_inference(
            predictor=predictor,
            base_video_dir=args.base_video_dir,
            input_mask_dir=args.input_mask_dir,
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            input_frame_inds = [initial_prompt],  # [initial_prompt]
            gt=gt_list,
            score_thresh=args.score_thresh,
            use_all_masks=args.use_all_masks,
            per_obj_png_file=args.per_obj_png_file,
            save_palette_png=args.save_palette_png,
            prompt=args.prompt,
            )
        
 
        
            
        
        if video_segments_first != None:
            binary_masks_first = []
            for frame_index, segment in video_segments_first.items():
                mask = segment[1]  # Assuming segment is a tuple of (frame_index, mask)
                binary_mask = (mask > 0).astype(np.uint8)  # Convert to binary mask with values 0 and 1
                binary_masks_first.append(binary_mask)
                  
        else:
            binary_masks_first = None
            
   
        L_no_defer, L_no_defer_sam_loss, L_no_defer_focal_loss = compute_downstream_loss(binary_masks_first, gt_list, frame_indices_for_clip)
        
        clip_frames = []
        for idx in frame_indices_for_clip:
            frame_mask = video_segments_first[idx][1]
            frame_mask = np.squeeze(frame_mask)  
            frame_mask = Image.fromarray(frame_mask)
            frame_mask = timesformer_transform(frame_mask)
            clip_frames.append(frame_mask)
                                
        clip = torch.stack(clip_frames, dim=1)
        
        
        
        # -------------------Correction Prompts -------------------------------------
        
        for second_prompt in range (initial_prompt+1, len(frame_names)):
            

            print("second_prompt: ", second_prompt)
            logging.info("second_prompt: " + str(second_prompt))
        
            input_frame_inds = [initial_prompt, second_prompt]
            
            folder_name = "_".join(map(str, input_frame_inds))
            folder_name_list.append(folder_name)
            output_mask_dir = os.path.join(args.output_mask_dir, video_name, folder_name)
            
            video_segments_cor, confidence_scores_cor = vos_inference(
                predictor=predictor,
                base_video_dir=args.base_video_dir,
                input_mask_dir=args.input_mask_dir,
                output_mask_dir=output_mask_dir,
                video_name=video_name,
                input_frame_inds = input_frame_inds,
                gt=gt_list,
                score_thresh=args.score_thresh,
                use_all_masks=args.use_all_masks,
                per_obj_png_file=args.per_obj_png_file,
                save_palette_png=args.save_palette_png,
                prompt=args.prompt,
                )
                   
                         
                
            
            binary_masks_cor = []
            
            if video_segments_cor is None:
                binary_masks_cor = None
            else:
                for frame_index, segment in video_segments_cor.items():
                    mask = segment[1]  # Assuming segment is a tuple of (frame_index, mask)
                    binary_mask = (mask > 0).astype(np.uint8)  # Convert to binary mask with values 0 and 1
                    binary_masks_cor.append(binary_mask)

            # Corrected downstream loss
            L_post_defer, L_post_defer_sam_loss, L_post_defer_focal_loss = compute_downstream_loss(binary_masks_cor, gt_list, frame_indices_for_clip)
            
            
            L_post_defer_list.append(L_post_defer)
            L_post_defer_sam_loss_list.append(L_post_defer_sam_loss)
            L_post_defer_focal_loss_list.append(L_post_defer_focal_loss)
            # ll_post_dice.append(L_post_defer_dice_loss_list)
            # ll_post_sam.append(L_post_defer_sam_loss_list_cor)
            # ll_post_focal.append(L_post_defer_focal_loss_list_cor)

             
      
                
        # Save all lists in a single file
        data_pkl_folder = os.path.join(args.post_hoc_model_save_dir, "data_pkl")
        os.makedirs(data_pkl_folder, exist_ok=True)
        with open(os.path.join(data_pkl_folder,f'{video_name}_data.pkl'), 'wb') as f:
            pickle.dump({'video_name':video_name, 'Masks':clip, 'L_no_defer':L_no_defer, 'L_post_defer_list':L_post_defer_list, 'L_post_defer_sam_loss_list':L_post_defer_sam_loss_list, 'L_no_defer_sam_loss':L_no_defer_sam_loss, 'L_no_defer_focal_loss':L_no_defer_focal_loss, 'L_post_defer_focal_loss_list':L_post_defer_focal_loss_list}, f)
            
        break
   
                
    
    print(f"completed inference on {len(video_names)} videos -- output masks saved to {args.output_mask_dir}")
    logging.info(f"completed inference on {len(video_names)} videos -- output masks saved to {args.output_mask_dir}")


if __name__ == "__main__":
    main()