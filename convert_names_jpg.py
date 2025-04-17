# convert names to 000001.jpg

import os
from pathlib import Path
import cv2

def convert_image_filenames(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding output directory
        rel_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Process each file
        for index, file in enumerate(files):
            
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_new = int(file[:-4])
                print (file_new) 
                input_path = os.path.join(root, file)
                # Format the output filename with leading zeros
                output_filename = f"{file_new:05d}.jpg"  # 5-digit format
                output_path = os.path.join(output_subdir, output_filename)
                
                # Read and convert image
                img = cv2.imread(input_path)
                if img is not None:
                    cv2.imwrite(output_path, img)
                    print(f"Converted: {input_path} -> {output_path}")
                else:
                    print(f"Failed to read: {input_path}")

if __name__ == "__main__":
    input_dir = "media/image"
    output_dir = "media/image_jpg"
    convert_image_filenames(input_dir, output_dir)