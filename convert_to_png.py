import os
import cv2
import numpy as np
from pathlib import Path

def count_object_pixels(mask_path):
    # Read the mask image
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"Failed to read: {mask_path}")
        return None
    
    # Handle different mask dimensions
    if mask.ndim == 3:
        # RGB mask → binary mask
        mask = np.any(mask != 0, axis=-1)
    elif mask.ndim == 2:
        # Already 2D, just ensure it's boolean
        mask = mask != 0
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    
    # Get unique object IDs (excluding 0 which is background)
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    
    # Count pixels for each object ID
    pixel_counts = {}
    for obj_id in object_ids:
        pixel_counts[obj_id] = np.sum(mask == obj_id)
    
    return pixel_counts

def convert_images_to_png(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding output directory
        rel_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Set the number of digits for padding to 5
        max_digits = 5  # Fixed padding length
        
        # Process each file
        for index, file in enumerate(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                file_new = int(file[:-4])
                print (file_new)
                # Format the output filename with leading zeros
                output_filename = f"{file_new:0{max_digits}}.png"
                output_path = os.path.join(output_subdir, output_filename)
                
                # Read and convert image
                img = cv2.imread(input_path)
                if img is not None:
                    cv2.imwrite(output_path, img)
                    print(f"Converted: {input_path} -> {output_path}")
                    
                    # Count object pixels
                    pixel_counts = count_object_pixels(input_path)
                    if pixel_counts:
                        print(f"Object pixel counts for {file}:")
                        for obj_id, count in pixel_counts.items():
                            print(f"  Object ID {obj_id}: {count} pixels")
                else:
                    print(f"Failed to read: {input_path}")

if __name__ == "__main__":
    input_dir = "media/mask"
    output_dir = "media/mask_png"
    convert_images_to_png(input_dir, output_dir) 