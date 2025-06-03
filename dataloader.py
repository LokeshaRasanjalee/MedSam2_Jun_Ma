# dataset.py
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
import os
import glob
import gc
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import InterpolationMode
from functools import lru_cache

class ClipDataset(Dataset):
    def __init__(self, pickle_file, args):
        # Get npz directory one step back
        self.npz_dir = os.path.join(os.path.dirname(os.path.dirname(pickle_file)), 'data_npz_4')
        self.args = args
        self.npz_files = sorted(glob.glob(os.path.join(self.npz_dir, '*.npz')))
        
        # Store only file paths and video names
        self.video_metadata = []
        
        for npz_file in self.npz_files:
            data = np.load(npz_file)
                 
            self.video_metadata.append({
                'npz_file': npz_file,
                # 'masks': torch.from_numpy(data['masks']),
                # 'no_df_sam_complement': torch.from_numpy(data['no_df_sam_complement']),
                # 'post_df_sam_complement': torch.from_numpy(data['post_df_sam_complement'])
            })
                
               
                
            if not self.args.full_run and len(self.video_metadata) >= 64:
                break   
                
        print(f"Loaded metadata for {len(self.video_metadata)} videos.")

    # @lru_cache(maxsize=1000)  # Keep last 1000 files in cache for maximum speed
    # def load_pickle_data(self, pickle_file):
    #     """Cache the pickle file data to avoid repeated disk reads."""
    #     with open(pickle_file, 'rb') as f:
    #         return pickle.load(f)

    def __len__(self):
        return len(self.video_metadata)

    def __getitem__(self, idx):
        info = self.video_metadata[idx]
        data = np.load(info['npz_file'])
        
        clips_batch = torch.from_numpy(data['masks']).permute(1, 0, 2, 3)

        # Reshape to [frames * channels, H, W] = [20, 512, 512]
        masks = clips_batch.reshape(-1, 512, 512)
        
        
        # return (info['masks'],
        #         info['no_df_sam_complement'],
        #         info['post_df_sam_complement'],
        #         info['npz_file'])
        
        return (
            masks,
            torch.from_numpy(data['no_df_sam_complement']),
            torch.from_numpy(data['post_df_sam_complement']),
            info['npz_file']
        )


def get_dataloaders(pickle_file_folder, args, batch_size=8, split_ratio=0.8):
    dataset = ClipDataset(pickle_file_folder, args)

    # Simple random split instead of stratified split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))
    
    # Shuffle indices
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    # Optimized DataLoader configuration for speed
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,  # Increased workers for faster loading
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,  # Increased prefetch for better throughput
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader