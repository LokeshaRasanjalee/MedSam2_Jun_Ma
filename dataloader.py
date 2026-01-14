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
        test_dataset = None  # Not available in this mode
    else:
        dataset = ClipDataset(args.data_npz_dir, args)
        npz_files = dataset.npz_files
        
        # Load the split dictionary
        with open(args.split_dict_path, 'r') as f:
            data_split_dict = eval(f.read())  # Load the dictionary from file
        
        # Get the subject IDs for train (0), val (1), and test (2) sets
        if 0 not in data_split_dict or 1 not in data_split_dict or 2 not in data_split_dict:
            raise ValueError(f"Split dictionary must contain keys 0 (train), 1 (val), and 2 (test). Available keys: {list(data_split_dict.keys())}")
        
        # Convert to sets of strings for comparison (split_dict has string IDs)
        train_subject_ids = set(str(id) for id in data_split_dict[0])
        val_subject_ids = set(str(id) for id in data_split_dict[1])
        test_subject_ids = set(str(id) for id in data_split_dict[2])
        
        print(f"Train subject IDs: {sorted(train_subject_ids)}")
        print(f"Validation subject IDs: {sorted(val_subject_ids)}")
        print(f"Test subject IDs: {sorted(test_subject_ids)}")
        
        # Extract subject IDs from npz file names and create train/val/test indices
        train_idx = []
        val_idx = []
        test_idx = []
        
        for i, npz_file in enumerate(npz_files):
            # Extract subject ID from filename - it's the part before the first underscore
            filename = os.path.basename(npz_file)
            # Get the part before the first underscore as subject ID
            subject_id = filename.split('_')[0]
            
            if subject_id in train_subject_ids:
                train_idx.append(i)
            elif subject_id in val_subject_ids:
                val_idx.append(i)
            elif subject_id in test_subject_ids:
                test_idx.append(i)
            else:
                print(f"Warning: Subject ID {subject_id} from filename {filename} not found in any split. Adding to train set.")
                train_idx.append(i)
        
        print(f"Train indices: {len(train_idx)}, Validation indices: {len(val_idx)}, Test indices: {len(test_idx)}")

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)
    
    # Check max index distribution
    train_dist = get_min_index_distribution(args, train_dataset)
    val_dist = get_min_index_distribution(args, val_dataset)

    print("Train split distribution:", dict(train_dist))
    print("Validation split distribution:", dict(val_dist))
    
    if test_dataset is not None:
        test_dist = get_min_index_distribution(args, test_dataset)
        print("Test split distribution:", dict(test_dist))

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
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    return train_loader, val_loader, test_loader