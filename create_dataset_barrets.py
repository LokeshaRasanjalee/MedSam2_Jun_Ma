import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import re

# Define source directories
IMAGE_DIR = "/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/Barretts_data/nbi_masks/from_videos/frames"
MASK_DIR = "/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/Barretts_data/nbi_masks/from_videos/masks"

# Create output directories
OUTPUT_DIR = "/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/Barretts_data/dataset2_100_1_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Masks"), exist_ok=True)

# Parameters
len_video = 100 # Lenght of the video clip
frame_interval = 1 #  gap between frames within a clip
inter_clip_stride = 2  # How far to move to start the next clip

def has_object(mask_path):
    """Check if mask contains any non-zero pixels"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask is not None and np.any(mask > 0)

def get_frame_number(filename):
    """Extract frame number from filename like 'case_M_20181001100941_0U62372100109341_1_005_001-1_a2_ayy_image0001.jpg'"""
    # Split by underscore and get the last part
    last_part = filename.split('_')[-1]
    # Remove 'image' and get the number
    frame_number = last_part.replace('image', '')
    # Remove extension
    frame_number = frame_number.split('.')[0]
    return frame_number

def get_sequence_id(filename):
    """Extract sequence ID (a<number>) from filename"""
    match = re.search(r'_a(\d+)_', filename)
    if match:
        return match.group(1)
    return None

# Get all mask files, filter out hidden files, and sort by index at end of filename
def get_index_from_filename(filename):
    """Extract the index number from the end of filename like 'text_text_.._<index>.jpg'"""
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]
    # Split by underscore and get the last part
    last_part = name_without_ext.split('_')[-1]
    try:
        return int(last_part)
    except ValueError:
        return 0  # Default to 0 if can't parse

def process_dataset():
    # Get list of mask folders
    mask_folders = [f for f in os.listdir(MASK_DIR) if os.path.isdir(os.path.join(MASK_DIR, f))]
    print(f"Found {len(mask_folders)} folders in {MASK_DIR}")
    mask_folders.sort()
    
    for mask_folder in mask_folders:
        mask_folder_path = os.path.join(MASK_DIR, mask_folder)
        
        
        image_folder_name = mask_folder
            
        image_folder_path = os.path.join(IMAGE_DIR, image_folder_name)
        
        if not os.path.exists(image_folder_path):
            print(f"Warning: No image folder found for {mask_folder}")
            continue
        
  
        
        # Filter out files starting with '.' and get image files
        all_files = [f for f in os.listdir(mask_folder_path) if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
        
        # Sort by the index at the end of the filename
        mask_files = sorted(all_files, key=get_index_from_filename)
        
        seq_files = mask_files
        
        # # Group files by sequence ID
        # sequence_groups = {}
        # for mask_file in mask_files:
        #     seq_id = get_sequence_id(mask_file)
        #     if seq_id:
        #         if seq_id not in sequence_groups:
        #             sequence_groups[seq_id] = []
        #         sequence_groups[seq_id].append(mask_file)
        
        # Process each sequence group
        # for seq_id, seq_files in sequence_groups.items():
            # print(f"\nProcessing sequence {seq_id} with {len(seq_files)} files")
        # seq_files.sort()  # Sort files within sequence
        
        # Find sequences of frames with objects spaced by frame_interval
        valid_sequences = []
        
        # Try starting from each possible frame in this sequence, with inter_clip_stride
        for start_idx in range(0, len(seq_files), inter_clip_stride):
            current_sequence = []
            current_idx = start_idx
            
            # Try to get len_video frames with frame_interval spacing
            while len(current_sequence) < len_video and current_idx < len(seq_files):
                mask_path = os.path.join(mask_folder_path, seq_files[current_idx])
                
                if has_object(mask_path):
                    current_sequence.append(current_idx)
                
                current_idx += frame_interval
            
            # If we found a complete sequence, add it
            if len(current_sequence) == len_video:
                valid_sequences.append(current_sequence)
        
        print(f"Found {len(valid_sequences)} valid sequences in sequence {mask_folder}")
        
        # Process each valid sequence
        for seq_idx, sequence in enumerate(valid_sequences):
            # Get the first image's frame number for folder naming
            first_mask_file = seq_files[sequence[0]]
            first_image_file = first_mask_file.replace('.png', '.jpg')
            folder_frame_number = get_frame_number(first_image_file)
            
            # Create sequence folders
            seq_folder_name = f"{mask_folder}_{folder_frame_number}"
            seq_image_folder = os.path.join(OUTPUT_DIR, "Images", seq_folder_name)
            seq_mask_folder = os.path.join(OUTPUT_DIR, "Masks", seq_folder_name)
            
            os.makedirs(seq_image_folder, exist_ok=True)
            os.makedirs(seq_mask_folder, exist_ok=True)
            
            # Copy files for this sequence
            for i, frame_idx in enumerate(sequence):
                mask_file = seq_files[frame_idx]
                image_file = mask_file.replace('.png', '.jpg')
                
                # Create new filenames with sequential numbering
                new_number = f"{i:04d}"  # Format as 0000, 0001, etc.
                new_mask_filename = f"{new_number}.png"
                new_image_filename = f"{new_number}.jpg"
                
                # Copy mask
                src_mask = os.path.join(mask_folder_path, mask_file)
                dst_mask = os.path.join(seq_mask_folder, new_mask_filename)
                shutil.copy2(src_mask, dst_mask)
                
                # Copy image
                src_image = os.path.join(image_folder_path, image_file)
                dst_image = os.path.join(seq_image_folder, new_image_filename)
                shutil.copy2(src_image, dst_image)
            
            print(f"Created sequence folder {seq_folder_name} with {len(sequence)} frames")
            
        print(f"Finished processing folder {mask_folder}")
        # break

if __name__ == "__main__":
    print("Starting dataset creation...")
    process_dataset()
    print("Dataset creation completed!")
