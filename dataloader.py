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
        
        self.video_metadata = []  # list of (video_path, frame_names, no_df_dice, post_df_dice, video_name)
        
        self.loss_min_sam_loss = 1
        self.loss_max_sam_loss = 0
        # self.loss_min_dice = 1
        # self.loss_max_dice = 0

        for file in glob.glob(os.path.join(pickle_file, '*.pkl')):
            with open(file, 'rb') as f:
                #print(f"Loading {file}")
                data = pickle.load(f)
                
                if len(data['L_post_defer_sam_loss_list']) != 4:
                    continue

                video_name = data['video_name']
                video_path = os.path.join(args.base_video_dir, video_name)
                frame_list = sorted(os.listdir(video_path))
                data['Masks'] = self.mask_transform(data['Masks'])
                
                min_vals = data['Masks'].amin(dim=[-2, -1], keepdim=True)
                max_vals = data['Masks'].amax(dim=[-2, -1], keepdim=True)
                normalized = (data['Masks'] - min_vals) / (max_vals - min_vals + 1e-6)
                data['Masks'] = normalized
                
                # data['L_no_defer'] = 1-data['L_no_defer']
                # data['L_post_defer_list'] = [1-x for x in data['L_post_defer_list']]

                self.video_metadata.append({
                    'video_path': video_path,
                    'frame_list': frame_list,
                    'L_no_defer_sam_loss': data['L_no_defer_sam_loss'],
                    'L_post_defer_sam_loss_list': data['L_post_defer_sam_loss_list'],
                    'video_name': video_name,
                    'masks': data['Masks']
                })
                # print(f"video: {video_name}, {data['L_no_defer']}, {data['L_post_defer_list']}")
                all_losses_sam_loss = [data['L_no_defer_sam_loss']] + data['L_post_defer_sam_loss_list']
                #all_losses_dice = [data['L_no_defer']] + data['L_post_defer_list']
                all_losses_sam_loss = np.array(all_losses_sam_loss)
                #all_losses_dice = np.array(all_losses_dice)
                #all_losses = torch.cat(all_losses, dim=0)
                if all_losses_sam_loss.min() < self.loss_min_sam_loss:
                    self.loss_min_sam_loss = all_losses_sam_loss.min()
                if all_losses_sam_loss.max() > self.loss_max_sam_loss:
                    self.loss_max_sam_loss = all_losses_sam_loss.max()
                    # print(f"video: {video_name}, Loss max: {self.loss_max}")
                # if all_losses_dice.min() < self.loss_min_dice:
                #     self.loss_min_dice = all_losses_dice.min()
                # if all_losses_dice.max() > self.loss_max_dice:
                #     self.loss_max_dice = all_losses_dice.max()
                    
                # if all_losses.max() > 1:
                #     print(f"video: {video_name}, Loss greater than 1: {all_losses.max()}")
                
                
                
                
                del data
                gc.collect()
                if len(self.video_metadata) >= 64:
                    break
        
        print("Loaded metadata only.")
        print(f"Loss min: {self.loss_min_sam_loss}, Loss max: {self.loss_max_sam_loss}")
        # print(f"Loss min: {self.loss_min_dice}, Loss max: {self.loss_max_dice}")

    def __len__(self):
        return len(self.video_metadata)

    def __getitem__(self, idx):
        info = self.video_metadata[idx]
        # video_frames = []
        # for frame_name in info['frame_list']:
        #     frame_path = os.path.join(info['video_path'], frame_name)
        #     frame = Image.open(frame_path)
        #     video_frames.append(self.transform(frame))

        # video_tensor = torch.stack(video_frames)
        
        masks = info['masks']
        masks = masks.permute(1, 0, 2, 3)  #((B, T, C, H, W))
        
        
        
        #combined_clip = torch.cat([video_tensor, masks], dim=1)
        
        L_no_defer_sam_loss = torch.tensor(info['L_no_defer_sam_loss'], dtype=torch.float32)
        L_post_defer_sam_loss_list = torch.tensor(info['L_post_defer_sam_loss_list'], dtype=torch.float32)

        # Min-max normalization: (x - min) / (max - min)
        denom = self.loss_max_sam_loss
        no_df_sam_loss_norm = (L_no_defer_sam_loss) / denom
        post_df_sam_loss_norm = (L_post_defer_sam_loss_list) / denom
        
        no_df_sam_complement = 1-no_df_sam_loss_norm
        post_df_sam_complement = 1-post_df_sam_loss_norm
        
        # print(f"no_df_dice: {no_df_dice_norm}, post_df_dice: {post_df_dice_norm}")
        
        
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