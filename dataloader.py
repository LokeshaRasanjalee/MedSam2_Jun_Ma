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

class ClipDataset(Dataset):
    def __init__(self, pickle_file, args):
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # RGB means
                         std=[0.229, 0.224, 0.225]) 
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),  # preserve class labels
        ])
        
        self.pickle_file = pickle_file
        self.args = args
        self.pickle_files = sorted(glob.glob(os.path.join(pickle_file, '*.pkl')))
        
        # Store only file paths and video names
        self.video_metadata = []
        for file in self.pickle_files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if len(data['L_post_defer_sam_loss_list']) != 4:
                    continue
                    
                video_name = data['video_name']
                video_path = os.path.join(args.base_video_dir, video_name)
                
                self.video_metadata.append({
                    'pickle_file': file,
                    'video_path': video_path,
                    'video_name': video_name
                })
                                
                del data
                gc.collect()
                # if len(self.video_metadata) >= 64:
                #     break
                
            
        
        print(f"Loaded metadata for {len(self.video_metadata)} videos.")

    def normalize_sample_losses(self, no_defer_loss, post_defer_losses):
        """Normalize losses within a single sample."""
        # Combine all losses for this sample
        all_losses = torch.cat([no_defer_loss.unsqueeze(0), post_defer_losses])
        
        # Get min and max for this sample
        min_loss = all_losses.min()
        max_loss = all_losses.max()
        
        # Normalize all losses
        normalized = (all_losses - min_loss) / (max_loss - min_loss + 1e-6)
        
        # Split back into no_defer and post_defer
        return normalized[0], normalized[1:]

    def __len__(self):
        return len(self.video_metadata)

    def __getitem__(self, idx):
        info = self.video_metadata[idx]
        
        # Load data on-demand
        with open(info['pickle_file'], 'rb') as f:
            data = pickle.load(f)
            
            # Get masks and losses
            masks = self.mask_transform(data['Masks'])
            L_no_defer_sam_loss = data['L_no_defer_sam_loss'].clone().detach().float()
            L_post_defer_sam_loss_list = torch.as_tensor(data['L_post_defer_sam_loss_list'], dtype=torch.float32).clone().detach()
            
            del data
            gc.collect()
        
        # Normalize masks
        min_vals = masks.amin(dim=[-2, -1], keepdim=True)
        max_vals = masks.amax(dim=[-2, -1], keepdim=True)
        masks = (masks - min_vals) / (max_vals - min_vals + 1e-6)
        masks = masks.permute(1, 0, 2, 3)  #((B, T, C, H, W))
        
        # Normalize losses within this sample
        no_df_sam_loss_norm, post_df_sam_loss_norm = self.normalize_sample_losses(
            L_no_defer_sam_loss, 
            L_post_defer_sam_loss_list
        )
        
        no_df_sam_complement = 1-no_df_sam_loss_norm
        post_df_sam_complement = 1-post_df_sam_loss_norm
        
        return (
            masks,
            no_df_sam_complement,
            post_df_sam_complement,
            info['video_name']
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

    train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # no need to shuffle validation
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader