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

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import csv
import logging
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

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

def check_cuda():
    """Check if CUDA is available and print device information."""
    if torch.cuda.is_available():
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("CUDA is not available. Using CPU.")
        return False


def main():
    # Start total runtime tracking
    total_start_time = time.time()
    parser = argparse.ArgumentParser()
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
        "-o",
        "--output_mask_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--data_pkl_dir",
        type=str,
        required=True,
        help="directory to save the data pkl",
    )
    
    parser.add_argument(
        "-e",
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment for logging and identification purposes",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)",
    )
    
    parser.add_argument(
        "--p1",
        type=float,
        default=234.0,
        help="P1 parameter for deferral loss (default: 312.0)",
    )
    parser.add_argument(
        "--p99",
        type=float,
        default=8384.0,
        help="P99 parameter for deferral loss (default: 10256.0)",
    )
    parser.add_argument(
        "--array_id",
        type=int,
        default=0,
        help = "array_id"
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        default=0,
        help = "number of groups"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="sam",
        help="loss type (default: sam)",
    )

    
    
    args = parser.parse_args()
    
    is_cuda_available = check_cuda()

  
    
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
    
   

    get_dataloaders(args.data_pkl_dir, args, batch_size=args.batch_size)

    


if __name__ == "__main__":
    main()