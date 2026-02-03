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
        elif args.loss_type == "iou":
            self.npz_dir = os.path.join(os.path.dirname(os.path.dirname(pickle_file)), 'data_npz_4_iou')
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
                
                if len(image_files) > 0:
                    # Calculate gap, ensuring it's at least 1 to avoid ValueError in range()
                    # if args.sample_factor is larger than len(image_files) or 0.
                    # Assuming args.sample_factor is a positive integer.
                    gap = len(image_files) // args.sample_factor                    
                    prompt_frames = list(range(0, len(image_files), gap))
                    
                    # Filter image_files to include only the selected frames
                    image_files_prompted = [image_files[i] for i in prompt_frames]
                
                if not image_files_prompted:
                    continue
                
                # Load and transform images
                images = []
                for img_path in image_files_prompted:
                    img = Image.open(img_path).convert('RGB')
                    img = self.image_transform(img)
                    images.append(img)
                
                # Stack images into a tensor
                images = torch.stack(images)  # Shape: [T, C, H, W]
                images = images.permute(1, 0, 2, 3)
                
                if self.args.loss_type == "sam":
                    Loss_no_defer = np.array([data['L_no_defer_sam_loss']], dtype=object)  # single-element list
                    Loss_post_defer = np.array(data['L_post_defer_sam_loss_list'], dtype=object)
                elif self.args.loss_type == "dice":
                    Loss_no_defer = np.array([data['L_no_defer']], dtype=object)  # single-element list
                    Loss_post_defer = np.array(data['L_post_defer_list'], dtype=object)
                elif self.args.loss_type == "iou":
                    Loss_no_defer = np.array([data['L_no_defer']], dtype=object)  # single-element list
                    Loss_post_defer = np.array(data['L_post_defer_list'], dtype=object)

                # Concatenate: (N+1,) with first element no_defer
                all_losses = np.concatenate([Loss_no_defer, Loss_post_defer[1:]])
                
                #---Related to global loss calculation
                
                replaced_losses = np.array([
                    1 if (v is None or np.isnan(v)) else v
                    for v in all_losses
                ], dtype=np.float32)
                
                # Normalize using percentiles
                # Hardcorded percentiles to 0 and 1
                global_normalized = (replaced_losses - global_p1) / (global_p99 - global_p1)
                #global_normalized = (replaced_losses - global_p1) / (global_p99 - global_p1 + 1e-6)
                global_normalized = np.clip(global_normalized, 0, 1)
           
                
                global_no_df_loss_norm = global_normalized[0]
                global_post_df_loss_norm = global_normalized[1:]
                
                global_no_df_loss_complement = global_no_df_loss_norm
                global_post_df_loss_complement = global_post_df_loss_norm
                
                
                #---Related to loss calculation 
                
                
                # Normalize using min and max
                # Filter out None values first:
                # valid_losses = [v for v in all_losses if v is not None]

                # if len(valid_losses) == 0:
                #     raise ValueError(f"No valid losses found in all_losses for min/max calculation: {all_losses}")

                # # Compute min and max on valid floats only
                # min_local_loss = np.min(valid_losses)
                # max_local_loss = np.max(valid_losses)
                
                # local_replaced_losses = np.array([
                #     1 if (v is None or np.isnan(v)) else v
                #     for v in all_losses
                # ], dtype=np.float32) # to clamp values at 1
                
                # local_normalized = (local_replaced_losses - min_local_loss) / (max_local_loss - min_local_loss + 1e-6)
                # local_normalized = np.clip(local_normalized, 0, 1)  # Clamp values between 0 and 1
                
                # # if torch.sum(local_normalized == 1) > 1 or torch.sum(local_normalized == 0) > 1:
                # #     print(f"{video_name} : More than one 1 or 0 in local_normalized. {local_normalized}")
                # #     continue
                
                # local_no_df_loss_norm = local_normalized[0]
                # local_post_df_loss_norm = local_normalized[1:]
                
                # local_no_df_loss_complement =  local_no_df_loss_norm
                # local_post_df_loss_complement = local_post_df_loss_norm
                
                # local_no_df_loss_complement = 1 - local_no_df_loss_norm
                # local_post_df_loss_complement = 1 - local_post_df_loss_norm
                
                # Pre-compute normalized and permuted masks
                # masks = self.mask_transform(data['Masks'])
                masks = data['Masks']
                
                # masks: torch.Tensor of shape (1, 7, 112, 112)
                masks_np = masks.numpy()  # Convert to numpy for easy percentile computation
                
                masks_binary = masks_np > 0.5  # apply threshold for binary mask

                # Initialize array for normalized masks with same shape
                # normalized_masks = np.empty_like(masks_np)

                # for i in range(masks_np.shape[1]):  # iterate over 7 images
                #     img = masks_np[0, i]  # shape (112, 112)
                #     p1 = np.percentile(img, 1)
                #     p99 = np.percentile(img, 99)
                    
                #     norm_img = (img - p1) / (p99 - p1 + 1e-6)
                #     norm_img = np.clip(norm_img, 0, 1)
                    
                #     normalized_masks[0, i] = norm_img

                # # Convert back to torch.Tensor if needed
                # masks = torch.from_numpy(normalized_masks).float()
                
                images_np = images.numpy()
                combined = np.concatenate([masks_binary, images_np], axis=0)
                
                
                # Save data as npz file in the new directory
                base_name = os.path.basename(file)
                npz_file = os.path.join(self.npz_dir, base_name.replace('.pkl', '.npz'))

                # # Load corresponding IoU dict from sibling folder (../iou_dict)
                # base_folder = os.path.dirname(os.path.dirname(file))
                # if base_name.endswith('_data.pkl'):
                #     iou_base_name = base_name.replace('_data.pkl', '_iou_dict.pkl')
                # else:
                #     raise ValueError(f"Invalid base name: {base_name}")
                # iou_pkl_file = os.path.join(base_folder, 'iou_dict', iou_base_name)
                # with open(iou_pkl_file, 'rb') as f:
                #     iou_dict = pickle.load(f)
                #     machine_iou_list = iou_dict['0']
                #     # Iterate around the keys in the iou_dict except '0' and '0_0'
                #     defer_keys = [k for k in iou_dict.keys() if k not in ('0', '0_0')]
                #     diff_Lm_Ld_list = []
                #     for k in defer_keys:
                #         # INSERT_YOUR_CODE
                #         # Here k is like '0_5', we want the number part, i.e., 5 from '0_5'
                #         try:
                #             # Expecting k format '0_N', where N is number (e.g. '0_5')
                #             number = int(k.split('_')[1])  # catches ValueError if malformed
                #         except Exception as ex:
                #             raise ValueError(f"Key not in expected format '0_N': got {k}") from ex
                        
                #         machine_tail = machine_iou_list[number:]
                #         defer_tail = iou_dict[k][number:]

                #         # Complement each IoU value: x -> (1 - x)
                #         machine_tail_complement = [1 - v for v in machine_tail]
                #         defer_tail_complement = [1 - v for v in defer_tail]
                        
                #         # INSERT_YOUR_CODE
                #         machine_loss_mean = np.mean(machine_tail_complement)
                #         defer_loss_mean = np.mean(defer_tail_complement)
                #         diff = machine_loss_mean - defer_loss_mean
                #         diff_Lm_Ld_list.append(diff)
                      
                        
                
                
                
                np.savez(
                    npz_file,    
                    masks=combined,
                    # local_no_df_loss_complement=local_no_df_loss_complement,
                    # local_post_df_loss_complement=local_post_df_loss_complement,
                    global_no_df_loss_complement=global_no_df_loss_complement,
                    global_post_df_loss_complement=global_post_df_loss_complement,
                    # diff_Lm_Ld=diff_Lm_Ld_list,
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