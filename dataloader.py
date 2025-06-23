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
        self.image_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # RGB means
                         std=[0.229, 0.224, 0.225]) 
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((112, 112), interpolation=InterpolationMode.NEAREST),  # preserve class labels
        ])
        
        self.pickle_file = pickle_file
        self.args = args
        self.pickle_files = sorted(glob.glob(os.path.join(pickle_file, '*.pkl')))
        
        # Create npz directory one step outside of pickle directory
        if args.loss_type == "sam":
            self.npz_dir = os.path.join(os.path.dirname(os.path.dirname(pickle_file)), 'data_npz_4_sam')
        elif args.loss_type == "dice":
            self.npz_dir = os.path.join(os.path.dirname(os.path.dirname(pickle_file)), 'data_npz_4_dice')
        os.makedirs(self.npz_dir, exist_ok=True)
        
        global_p1 = args.p1
        global_p99 = args.p99
        
        if args.num_groups == 0:
            current_chunk = self.pickle_files
        else:
            num_groups = args.num_groups
            total = len(self.pickle_files)

            # Compute approximate chunk size
            chunked = []
            chunk_size = total // num_groups
            remainder = total % num_groups

            start = 0
            for i in range(num_groups):
                end = start + chunk_size + (1 if i < remainder else 0)
                chunked.append(self.pickle_files[start:end])
                start = end

            # Output number of elements in each chunk
            group_sizes = [len(group) for group in chunked]
            for i, size in enumerate(group_sizes):
                print(f"Group {i+1}: {size} items")

            # Output total
            print(f"\nTotal items across all groups: {sum(group_sizes)}")
            

            current_chunk = chunked[args.array_id] 
                
        
        
        # Store only file paths and video names
        self.video_metadata = []
        for file in current_chunk:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                # if len(data['L_post_defer_sam_loss_list']) != 4:
                #     continue
                    
                video_name = data['video_name']
                video_path = os.path.join(args.base_video_dir, video_name)
                
                # Load and sort images from the video folder
                image_files = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
                if not image_files:
                    continue
                
                # Load and transform images
                images = []
                for img_path in image_files:
                    img = Image.open(img_path).convert('RGB')
                    img = self.image_transform(img)
                    images.append(img)
                
                # Stack images into a tensor
                images = torch.stack(images)  # Shape: [T, C, H, W]
                images = images.permute(1, 0, 2, 3)
                
                # Pre-compute normalized losses and complements
                if self.args.loss_type == "sam":
                    Loss_no_defer = torch.tensor(data['L_no_defer_sam_loss'], dtype=torch.float32)
                    Loss_post_defer = torch.tensor(data['L_post_defer_sam_loss_list'], dtype=torch.float32)
                elif self.args.loss_type == "dice":
                    Loss_no_defer = torch.tensor(data['L_no_defer'], dtype=torch.float32)
                    Loss_post_defer = torch.tensor(data['L_post_defer_list'], dtype=torch.float32)
                
                # Normalize losses
                all_losses = torch.cat([Loss_no_defer.unsqueeze(0), Loss_post_defer])
                
                 # Normalize using percentiles
                global_normalized = (all_losses - global_p1) / (global_p99 - global_p1 + 1e-6)
                global_normalized = torch.clamp(global_normalized, 0, 1)  # Clamp values between 0 and 1
                
                # if torch.sum(global_normalized == 1) > 1 or torch.sum(global_normalized == 0) > 1:
                #     print(f"{video_name} : More than one 1 or 0 in global_normalized. {global_normalized}")
                #     continue
                
                global_no_df_loss_norm = global_normalized[0]
                global_post_df_loss_norm = global_normalized[1:]
                
                global_no_df_loss_complement = 1 - global_no_df_loss_norm
                global_post_df_loss_complement = 1 - global_post_df_loss_norm
                
                
                # Normalize using min and max
                min_local_loss = all_losses.min()
                max_local_loss = all_losses.max()
                local_normalized = (all_losses - min_local_loss) / (max_local_loss - min_local_loss + 1e-6)
                local_normalized = torch.clamp(local_normalized, 0, 1)  # Clamp values between 0 and 1
                
                # if torch.sum(local_normalized == 1) > 1 or torch.sum(local_normalized == 0) > 1:
                #     print(f"{video_name} : More than one 1 or 0 in local_normalized. {local_normalized}")
                #     continue
                
                local_no_df_loss_norm = local_normalized[0]
                local_post_df_loss_norm = local_normalized[1:]
                
                local_no_df_loss_complement = 1 - local_no_df_loss_norm
                local_post_df_loss_complement = 1 - local_post_df_loss_norm
                
                # Pre-compute normalized and permuted masks
                masks = self.mask_transform(data['Masks'])
                min_vals = masks.amin(dim=[-2, -1], keepdim=True)
                max_vals = masks.amax(dim=[-2, -1], keepdim=True)
                masks = (masks - min_vals) / (max_vals - min_vals + 1e-6)
                #masks = masks.permute(1, 0, 2, 3)  #((B, T, C, H, W))
                
                combined = torch.cat([masks,images], dim=0)
                
                
                # Save data as npz file in the new directory
                base_name = os.path.basename(file)
                npz_file = os.path.join(self.npz_dir, base_name.replace('.pkl', '.npz'))
                np.savez(
                    npz_file,    
                    masks=combined.numpy(),
                    local_no_df_loss_complement=local_no_df_loss_complement.numpy(),
                    local_post_df_loss_complement=local_post_df_loss_complement.numpy(),
                    global_no_df_loss_complement=global_no_df_loss_complement.numpy(),
                    global_post_df_loss_complement=global_post_df_loss_complement.numpy()
                )
                
                self.video_metadata.append(
                    npz_file,)
                
                del data
                gc.collect()
                
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
        
        return (
            info['masks'],
            info['no_df_sam_complement'],
            info['post_df_sam_complement'],
            info['video_name']
        )


def get_dataloaders(pickle_file_folder, args, batch_size=8, split_ratio=0.8):
    dataset = ClipDataset(pickle_file_folder, args)

    # # Simple random split instead of stratified split
    # dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # split = int(np.floor(split_ratio * dataset_size))
    
    # # Shuffle indices
    # np.random.seed(42)
    # np.random.shuffle(indices)
    
    # train_idx, val_idx = indices[:split], indices[split:]

    # train_dataset = Subset(dataset, train_idx)
    # val_dataset = Subset(dataset, val_idx)

    # # Optimized DataLoader configuration for speed
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=4,  # Increased workers for faster loading
    #     pin_memory=True,
    #     persistent_workers=True,
    #     prefetch_factor=2,  # Increased prefetch for better throughput
    #     drop_last=True
    # )

    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True,
    #     persistent_workers=True,
    #     prefetch_factor=2
    # )

    # return train_loader, val_loader