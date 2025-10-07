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
        self.npz_files = glob.glob(os.path.join(self.npz_dir, '*.npz'))
        
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
                
               
                
            if not self.args.full_run and len(self.video_metadata) >= 1000:
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
        
        #Here sholud be dice accuracy as no_df_loss_complement and post_df_loss_complement
        
        return (
            torch.from_numpy(data['masks']),
            torch.from_numpy(data['global_no_df_loss_complement']),
            torch.from_numpy(data['global_post_df_loss_complement']),
            os.path.basename(info['npz_file'])
        )
        
def get_min_index_distribution(args, dataset):
    counter = Counter()

    for i in tqdm(range(len(dataset))):
        _, no_df_val, post_df_vals, _ = dataset[i]  # Extract values

        # Ensure tensors are 1D
        no_df_val = no_df_val.view(-1)        # Shape: [1]
        post_df_vals = post_df_vals.view(-1)  # Shape: [9]
        post_df_vals = post_df_vals+args.alpha

        # Concatenate to get [10] vector
        combined = torch.cat([no_df_val, post_df_vals], dim=0)

        # Get index of max
        min_idx = int(torch.argmin(combined).item())

        # Count it
        counter[min_idx] += 1

    return counter


def get_dataloaders( args, batch_size=8, split_ratio=0.8):
    np.random.seed(42)
    if args.train_test_split:       
        train_dataset = ClipDataset(args.data_npz_dir_train, args)
        val_dataset = ClipDataset(args.data_npz_dir_test, args)
    else:
        dataset = ClipDataset(args.data_npz_dir, args)
        npz_files = dataset.npz_files
        
        # Load the split dictionary
        with open(args.split_dict_path, 'r') as f:
            data_split_dict = eval(f.read())  # Load the dictionary from file
        
        # Get the subject IDs for validation set from the specified array_id
        if args.array_id in data_split_dict:
            val_subject_ids = data_split_dict[args.array_id]
            print(f"Using array_id {args.array_id} with validation subject IDs: {val_subject_ids}")
        else:
            raise ValueError(f"array_id {args.array_id} not found in split dictionary. Available keys: {list(data_split_dict.keys())}")
        
        # Extract subject IDs from npz file names and create train/val indices
        train_idx = []
        val_idx = []
        
        for i, npz_file in enumerate(npz_files):
            # Extract subject ID from filename like "D_NBI_67_20160415_0_3_0_data.npz"
            filename = os.path.basename(npz_file)
            # Split by '_' and get the subject ID (3rd element after D_NBI)
            parts = filename.split('_')
            if len(parts) >= 3 and parts[0] == 'D' and parts[1] == 'NBI':
                subject_id = int(parts[2])
                
                if subject_id in val_subject_ids:
                    val_idx.append(i)
                else:
                    train_idx.append(i)
            else:
                print(f"Warning: Could not parse subject ID from filename: {filename}")
                # Default to training set if parsing fails
                train_idx.append(i)
        
        print(f"Train indices: {len(train_idx)}, Validation indices: {len(val_idx)}")
        print(f"Train subject IDs: {[int(os.path.basename(npz_files[i]).split('_')[2]) for i in train_idx]}")
        print(f"Validation subject IDs: {[int(os.path.basename(npz_files[i]).split('_')[2]) for i in val_idx]}")

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
    
    # Check max index distribution
    train_dist = get_min_index_distribution(args, train_dataset)
    val_dist = get_min_index_distribution(args, val_dataset)

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