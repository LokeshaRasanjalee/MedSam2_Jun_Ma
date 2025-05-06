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
import numpy as np
import seaborn as sns
from torchvision import transforms
from PIL import Image
import torch

def get_frame_names(video_dir):
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names = list(sorted(frame_names))
    return frame_names  

def get_mask_img_list_with_obj(args, frame_names, video_name):
    mask_img_list = [
        name
        for idx, name in enumerate(frame_names)
        if os.path.exists(
            os.path.join(args.input_mask_dir, video_name, f"{name}.png")
        )
    ]
    mask_img_list_with_obj = sorted([
        idx
        for idx, name in enumerate(mask_img_list)
        if np.any(np.array(Image.open(os.path.join(args.input_mask_dir, video_name, f"{name}.png")).convert('L')) > 0)
    ])

    return mask_img_list_with_obj

class ClipDataset(Dataset):
    def __init__(self,args, pickle_file, save_dir):
        
        # Define the transform for RGB images
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Mask transform (grayscale only, no normalization)
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # produces 1x224x224
        ])
      

        self.clips = []
        self.delta_Ls=[]
        self.L_post_defer_full_list=[]
        self.labels =[]
        self.mask_list=[]
        self.confidence = []
        
        for file in glob.glob(os.path.join(pickle_file, '*.pkl')):
            seq_name = file.split('/')[-1].split('_')[0].split('data')[0]
            print ("Sequence: ", seq_name)
            
            if seq_name != 'seq14':
                continue
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if len(data['clips'])==0:
                    continue
                
                processed_clips = []

                for clip in data['clips_without_trans']:
                    transformed_frames = []
                    
                    for t in range(clip.shape[1]):  # assuming shape is (H, T, W)
                        frame = clip[:, t, :]  # shape: (H, W)
                        frame = frame.numpy().astype('uint8')  # convert to numpy uint8
                        #frame = np.stack((frame,)*3, axis=-1)  # convert to 3 channels
                        frame = Image.fromarray(frame)  # convert to PIL Image

                        transformed_frame = self.mask_transform (frame)  # now shape (3, 224, 224)
                        transformed_frames.append(transformed_frame)

                    # Stack back: (T, C, H, W)
                    clip_tensor = torch.stack(transformed_frames, dim=0)
                    processed_clips.append(clip_tensor)

                # If needed, batch them: (B, T, C, H, W)
                mask_stack = torch.stack(processed_clips)
                #self.clips.extend(clip_stack)
                
                
                #Images
                processed_clips = []

                for clip in data['img_frame_list']:
                    transformed_frames = []
                    
                    for t in range(len(clip)):  # assuming shape is (H, T, W)
                        frame = clip[t]  # shape: (H, W)
                        #frame = frame.numpy().astype('uint8')  # convert to numpy uint8
                        #frame = np.stack((frame,)*3, axis=-1)  # convert to 3 channels
                        frame = Image.fromarray(frame)  # convert to PIL Image

                        transformed_frame = self.rgb_transform(frame)  # now shape (3, 224, 224)
                        transformed_frames.append(transformed_frame)

                    # Stack back: (T, C, H, W)
                    clip_tensor = torch.stack(transformed_frames, dim=0)
                    processed_clips.append(clip_tensor)

                img_stack = torch.stack(processed_clips)
                combined_clip = torch.cat([mask_stack, img_stack], dim=2)
                self.clips.extend(combined_clip)
                
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
                
                
                
                mask_frame_list = []
                frame_names = get_frame_names(os.path.join(args.base_video_dir, seq_name))    
                mask_img_list_with_obj = sorted(get_mask_img_list_with_obj(args, frame_names, seq_name))
                initial_prompt = int(mask_img_list_with_obj[0])
                mask_img_list_with_obj.pop(0)
                
                half_window = 4
                
                for second_prompt in range (initial_prompt+1, len(frame_names)):
            
                    if (second_prompt >= initial_prompt + half_window) and (second_prompt < len(frame_names) - half_window):
                        # GOOD → continue normal processing
                        pass
                    else:
                        continue  # Skip this second_promptsecond_prompt >=initial_prompt+half_window)) or (second_prompt < (len(frame_names)-half_window)):
                    if second_prompt in mask_img_list_with_obj:
                        mask_frame_list.append(second_prompt)
                
                self.mask_list.extend(mask_frame_list) 
                
                
                
                
                
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
        frame_idx = self.mask_list[idx]
        return clip, torch.tensor(label, dtype=torch.float32), delta_l,frame_idx

    def count_labels(self):
        count_1s = sum(1 for label in self.labels if label == 1)
        count_0s = len(self.labels) - count_1s
        return count_1s, count_0s

def get_dataloaders(args, pickle_file_folder, save_dir, batch_size=8, split_ratio=0.8):
    dataset = ClipDataset(args, pickle_file_folder, save_dir)
    # labels = [data[1] for data in dataset]  # Assuming the dataset returns (input, label)

    # stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=1-split_ratio, random_state=42)
    # train_idx, val_idx = next(stratified_split.split(range(len(dataset)), labels))

    # train_dataset = Subset(dataset, train_idx)
    # val_dataset = Subset(dataset, val_idx)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return val_loader, val_loader