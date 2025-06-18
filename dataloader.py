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
from collections import Counter
from tqdm import tqdm

class ClipDataset(Dataset):
    def __init__(self, pickle_file, args):
        # Get npz directory one step back
        self.npz_dir = pickle_file
        self.args = args
        self.npz_files = sorted(glob.glob(os.path.join(self.npz_dir, '*.npz')))
        
        # Store only file paths and video names
        self.video_metadata = []
        
        for npz_file in self.npz_files:
            data = np.load(npz_file)
            
            no_df_dice_batch = torch.from_numpy(data['global_no_df_sam_complement'])
            post_df_dice_batch = torch.from_numpy(data['global_post_df_sam_complement'])
            
            all_costs = torch.cat([1-no_df_dice_batch.unsqueeze(0), (1-post_df_dice_batch)+args.beta], dim=0)
            
            adjusted_costs = (1-post_df_dice_batch) + args.beta
            # All possible accuracies: base + n_e frames with adjusted gain
            all_cost_adjusted = torch.cat([1-no_df_dice_batch.unsqueeze(0), adjusted_costs], dim=0)
            # Best cost (oracle) using argmin on adjusted gains
            best_actions = torch.argmin(all_cost_adjusted, dim=0)
            best_cost = torch.gather(all_costs, 0, best_actions.unsqueeze(0)).squeeze(0)

            
                 
            self.video_metadata.append({
                'npz_file': npz_file,
                'best_accs': 1-best_cost,
                'best_actions': best_actions,
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
        
        # return (info['masks'],
        #         info['no_df_sam_complement'],
        #         info['post_df_sam_complement'],
        #         info['npz_file'])
        
        return (
            torch.from_numpy(data['masks']),
            torch.from_numpy(data['global_no_df_sam_complement']),
            torch.from_numpy(data['global_post_df_sam_complement']),
            info['npz_file'],
            info['best_accs'],
            info['best_actions']
        )
        
def get_max_index_distribution(dataset):
    counter = Counter()

    for i in tqdm(range(len(dataset))):
        _, no_df_val, post_df_vals, _, _, _ = dataset[i]  # Extract values

        # Ensure tensors are 1D
        no_df_val = no_df_val.view(-1)        # Shape: [1]
        post_df_vals = post_df_vals.view(-1)  # Shape: [9]

        # Concatenate to get [10] vector
        combined = torch.cat([no_df_val, post_df_vals], dim=0)

        # Get index of max
        max_idx = int(torch.argmax(combined).item())

        # Count it
        counter[max_idx] += 1

    return counter


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
    
    # Check max index distribution
    train_dist = get_max_index_distribution(train_dataset)
    val_dist = get_max_index_distribution(val_dataset)

    print("Train split distribution:", dict(train_dist))
    print("Validation split distribution:", dict(val_dist))

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