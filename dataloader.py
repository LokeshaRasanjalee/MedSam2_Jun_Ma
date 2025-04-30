# dataset.py
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class ClipDataset(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        self.clips = data['clips']  # list of (C, T, H, W) tensors
        self.delta_Ls = data['delta_Ls']  # list of 0/1

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        delta_L = self.delta_Ls[idx]
        return clip, torch.tensor(delta_L, dtype=torch.float32)

def get_dataloaders(pickle_file, batch_size=8, split_ratio=0.8):
    dataset = ClipDataset(pickle_file)
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
