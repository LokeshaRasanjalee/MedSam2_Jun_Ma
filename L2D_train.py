# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from collections import defaultdict
import datetime

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
from dataloader import get_dataloaders
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

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

def dice_score(pred_mask, true_mask, eps=1e-5):
    #print ("dice_score")
    pred = pred_mask.flatten()
    true = true_mask.flatten()
    intersection = (pred * true).sum()
    return (2. * intersection) / (pred.sum() + true.sum() + eps)


def compute_frame_features(curr_mask, prev_mask, logit, confidence_score):
    #print ("compute_frame_features")
    dice = dice_score(curr_mask, prev_mask)
    conf_mean = logit.mean().item()
    conf_std = logit.std().item()
    area = curr_mask.sum().item()
    #edge_sharpness = compute_edge_sharpness(curr_mask)
    
    return [dice, conf_mean, conf_std, area, confidence_score]

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
    joblib.dump(model, os.path.join(output_path, model_name))
 
    
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
    total_iou = 0.0
    valid_frames = 0
    
    for idx in frame_indices_for_clip:
        if idx >= len(video_segments) or idx >= len(gt_list):
            continue  # Skip out of range indices

        pred_mask = video_segments[idx][1]  # Assuming your video_segments store (frame_index, mask) tuples
        gt_mask = gt_list[idx][0]            # Your gt_list stores (1, H, W) numpy arrays

        iou = compute_iou(pred_mask, gt_mask)
        total_iou += iou
        valid_frames += 1

    assert valid_frames != 0

    avg_iou = total_iou / valid_frames
    downstream_loss = 1.0 - avg_iou
    return downstream_loss

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for clips_batch, labels_batch in loader:
        clips_batch = clips_batch.to(device)
        labels_batch = labels_batch.to(device)

        clips_batch = clips_batch.repeat(1, 3, 1, 1, 1)
        outputs = model(clips_batch).squeeze(1)
        loss = criterion(outputs, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clips_batch.size(0)

        # Compute accuracy
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels_batch).sum().item()
        total += labels_batch.size(0)

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for clips_batch, labels_batch in loader:
            clips_batch = clips_batch.to(device)
            labels_batch = labels_batch.to(device)

            clips_batch = clips_batch.repeat(1, 3, 1, 1, 1)
            outputs = model(clips_batch).squeeze(1)
            loss = criterion(outputs, labels_batch)

            total_loss += loss.item() * clips_batch.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


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


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference_old(
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
    #os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
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
            
            # plt.figure(figsize=(9, 6))
            # plt.title(f"frame {input_frame_idx}")
            # plt.imshow(Image.open(os.path.join(base_video_dir, video_name, f"{frame_names[input_frame_idx]}.jpg")))
            # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
            # print(out_mask_logits.shape)
            
            # # Save the visualization image
            # vis_path = os.path.join(output_mask_dir, video_name, f"vis_{frame_names[input_frame_idx]}.png")
            # plt.savefig(vis_path)
            # plt.close()  # Close the figure to free memory

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
    

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_separate_inference_per_object(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
):
    """
    Run inference on a single video with the given predictor.

    Unlike `vos_inference`, this function run inference separately for each object
    in a video, which could be applied to datasets like LVOS or YouTube-VOS that
    don't have all objects to track appearing in the first frame (i.e. some objects
    might appear only later in the video).
    """
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
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    # collect all the object ids and their input masks
    inputs_per_object = defaultdict(dict)
    for idx, name in enumerate(frame_names):
        if per_obj_png_file or os.path.exists(
            os.path.join(input_mask_dir, video_name, f"{name}.png")
        ):
            per_obj_input_mask, input_palette = load_masks_from_dir(
                input_mask_dir=input_mask_dir,
                video_name=video_name,
                frame_name=frame_names[idx],
                per_obj_png_file=per_obj_png_file,
                allow_missing=True,
            )
            for object_id, object_mask in per_obj_input_mask.items():
                # skip empty masks
                if not np.any(object_mask):
                    continue
                # if `use_all_masks=False`, we only use the first mask for each object
                if len(inputs_per_object[object_id]) > 0 and not use_all_masks:
                    continue
                print(f"adding mask from frame {idx} as input for {object_id=}")
                inputs_per_object[object_id][idx] = object_mask


    # run inference separately for each object in the video
    object_ids = sorted(inputs_per_object)
    output_scores_per_object = defaultdict(dict)
    for object_id in object_ids:
        # add those input masks to SAM 2 inference state before propagation
        input_frame_inds = sorted(inputs_per_object[object_id])
        predictor.reset_state(inference_state)
        for input_frame_idx in input_frame_inds:
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=inputs_per_object[object_id][input_frame_idx],
            )

        # run propagation throughout the video and collect the results in a dict
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=min(input_frame_inds),
            reverse=False,
        ):
            obj_scores = out_mask_logits.cpu().numpy()
            output_scores_per_object[object_id][out_frame_idx] = obj_scores

    # post-processing: consolidate the per-object scores into per-frame masks
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_palette = input_palette or DAVIS_PALETTE

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for frame_idx in range(len(frame_names)):
        scores = torch.full(
            size=(len(object_ids), 1, height, width),
            fill_value=-1024.0,
            dtype=torch.float32,
        )
        for i, object_id in enumerate(object_ids):
            if frame_idx in output_scores_per_object[object_id]:
                scores[i] = torch.from_numpy(
                    output_scores_per_object[object_id][frame_idx]
                )

        if not per_obj_png_file:
            scores = predictor._apply_non_overlapping_constraints(scores)
        per_obj_output_mask = {
            object_id: (scores[i] > score_thresh).cpu().numpy()
            for i, object_id in enumerate(object_ids)
        }
        video_segments[frame_idx] = per_obj_output_mask

    # write the output masks as palette PNG files to output_mask_dir
    for frame_idx, per_obj_output_mask in video_segments.items():
        save_palette_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
            output_palette=output_palette,
        )
        
def calculate_auc(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predicted_probs = []

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in data_loader:
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
        for inputs, labels in data_loader:
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
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get model predictions
            inputs = inputs.repeat(1, 3, 1, 1, 1)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()  # Assuming binary classification

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds)

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    return cm

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
    args = parser.parse_args()

   
    
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
    
    print(f"Train on {len(video_names)} videos:\n{video_names}")
    logging.info(f"Train on {len(video_names)} videos:\n{video_names}")
    
    # ----- Prepare R(2+1)D model -----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.r2plus1d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Assuming binary classification (prompt vs no prompt)
    model = model.to(device)

    

    # ----- Define Resize Transform for R(2+1)D -----
    r2plus1d_transform = T.Compose([
        T.Resize((112, 112)),    # Downsample frames to 112x112
        T.ToTensor(),            # (H, W, C) -> (C, H, W)
    ])
    
    batch_size = 8
    num_epochs = 30
    learning_rate = 1e-5
    pickle_file = os.path.join(args.post_hoc_model_save_dir, 'data.pkl')
    
    train_loader, val_loader = get_dataloaders(pickle_file, batch_size=batch_size)

    # Calculate the number of positive and negative samples
    num_positive = sum(train_loader.dataset.dataset.labels)
    num_negative = len(train_loader.dataset.dataset) - num_positive

    # Calculate pos_weight
    pos_weight = num_negative / num_positive

    # Define the criterion with pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    
    #--------------------------Train Model----------------------------------
    
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} "
              f"Val Acc: {val_acc:.4f}")

    # Plotting Loss and Accuracy Curves
    plot_and_save_loss_accuracy_curves(train_losses, val_losses, train_accs, val_accs, args.output_mask_dir)

    # Calculate AUC
    auc = calculate_auc(model, val_loader, device)
    logging.info(f"AUC: {auc:.4f}")
    print(f"AUC: {auc:.4f}")

    # Plot ROC curve and save it
    plot_roc_curve(model, val_loader, device, args.output_mask_dir)

    # Calculate Confusion Matrix
    confusion_matrix = calculate_confusion_matrix(model, val_loader, device)
    logging.info(f"Confusion Matrix:\n{confusion_matrix}")
    print(f"Confusion Matrix:\n{confusion_matrix}")
    
    
    
    save_model(model, os.path.join(args.post_hoc_model_save_dir, f"{args.experiment_name}_{timestamp}"), "r_2_plus_1_d.pth")
    
   

    print(f"completed inference on {len(video_names)} videos -- output masks saved to {args.output_mask_dir}")
    logging.info(f"completed inference on {len(video_names)} videos -- output masks saved to {args.output_mask_dir}")


if __name__ == "__main__":
    main()