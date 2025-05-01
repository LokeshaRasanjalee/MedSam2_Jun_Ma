# dataset.py
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

class ClipDataset(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        self.clips = data['clips']  # list of (C, T, H, W) tensors
        self.labels = data['labels']  # list of 0/1
        self.delta_Ls = data['delta_Ls']

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        label = self.labels[idx]
        return clip, torch.tensor(label, dtype=torch.float32)

def get_dataloaders(pickle_file, batch_size=8, split_ratio=0.8):
    dataset = ClipDataset(pickle_file)
    labels = [data[1] for data in dataset]  # Assuming the dataset returns (input, label)

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=1-split_ratio, random_state=42)
    train_idx, val_idx = next(stratified_split.split(range(len(dataset)), labels))

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
