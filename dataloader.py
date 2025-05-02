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

class ClipDataset(Dataset):
    def __init__(self, pickle_file, save_dir):
      

        self.clips = []
        self.delta_Ls=[]
        self.L_post_defer_full_list=[]
        self.labels =[]
        self.confidence = []
        
        for file in glob.glob(os.path.join(pickle_file, '*.pkl')):
            print ("File: ",file)
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if len(data['clips'])==0:
                    continue
                self.clips.extend(data['clips'])
                
                #------For total_iou
                # self.L_no_defer_full_list = data['L_no_defer_full_list']
                # self.L_post_defer_full_list = data['L_post_defer_full_list']
                # delta_L = [a - b for a, b in zip(data['L_no_defer_full_list'], data['L_post_defer_full_list'])]
                # self.delta_Ls.extend(delta_L)
                # #delta_L = data['delta_Ls']
                # self.labels.extend([1 if delta_l > 0.28 else 0 for delta_l in delta_L])
                # self.confidence.extend(data['conf_list'])
                # del data
                # del delta_L
                # gc.collect()
                
                
                #------For regional_iou-----------
                self.delta_Ls.extend(data['delta_Ls'])
                #delta_L = data['delta_Ls']
                self.labels.extend([1 if delta_l > 0.7 else 0 for delta_l in data['delta_Ls']])
                self.confidence.extend(data['conf_list'])
                del data
                gc.collect()
                
                
        print ("Loaded all")
        # Plot the distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.delta_Ls, bins=30, kde=True)
        plt.title('Distribution of delta_Ls')
        plt.xlabel('delta_Ls')
        plt.ylabel('Frequency')

        # Save the plot
        plt.savefig(os.path.join(save_dir, 'delta_Ls_distribution.png'))
        plt.close() 

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        label = self.labels[idx]
        delta_l = self.delta_Ls [idx]
        conf = self.confidence[idx]
        return clip, torch.tensor(delta_l, dtype=torch.float32)

    def count_labels(self):
        count_1s = sum(1 for label in self.labels if label == 1)
        count_0s = len(self.labels) - count_1s
        return count_1s, count_0s

def get_dataloaders(pickle_file_folder, save_dir, batch_size=8, split_ratio=0.8):
    dataset = ClipDataset(pickle_file_folder, save_dir)
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader