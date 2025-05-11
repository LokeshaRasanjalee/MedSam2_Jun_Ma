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

class ClipDataset(Dataset):
    def __init__(self, pickle_file, args):
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])
        
        self.video_metadata = []  # list of (video_path, frame_names, no_df_dice, post_df_dice, video_name)
        save_dir = args.post_hoc_model_save_dir

        for file in glob.glob(os.path.join(pickle_file, '*.pkl')):
            with open(file, 'rb') as f:
                data = pickle.load(f)
                
                if len(data['L_post_defer_list']) != 4:
                    continue

                video_name = data['video_name']
                video_path = os.path.join(args.base_video_dir, video_name)
                frame_list = sorted(os.listdir(video_path))

                self.video_metadata.append({
                    'video_path': video_path,
                    'frame_list': frame_list,
                    'no_df_dice': data['L_no_defer'],
                    'post_df_dice': data['L_post_defer_list'],
                    'video_name': video_name,
                    'masks': data['Masks']
                })
                
                del data
                gc.collect()
        
        print("Loaded metadata only.")

    def __len__(self):
        return len(self.video_metadata)

    def __getitem__(self, idx):
        info = self.video_metadata[idx]
        video_frames = []
        for frame_name in info['frame_list']:
            frame_path = os.path.join(info['video_path'], frame_name)
            frame = Image.open(frame_path)
            video_frames.append(self.transform(frame))

        video_tensor = torch.stack(video_frames)
        masks = info['masks']
        masks = masks.permute(1, 0, 2, 3)  #((B, T, C, H, W))
        
        
        
        combined_clip = torch.cat([video_tensor, masks], dim=1)
        
        
        return (
            combined_clip,
            torch.tensor(info['no_df_dice'], dtype=torch.float32),
            torch.tensor(info['post_df_dice'], dtype=torch.float32),
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