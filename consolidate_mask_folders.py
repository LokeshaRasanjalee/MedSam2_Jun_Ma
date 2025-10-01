#!/usr/bin/env python3
"""
Script to consolidate all box_p0i_k10_* folders into a single box_p0i_k10_all folder.
This script will:
1. Create box_p0i_k10_all folder with data_pkl and iou_dict subfolders
2. Copy all contents from box_p0i_k10_*/data_pkl to box_p0i_k10_all/data_pkl
3. Copy all contents from box_p0i_k10_*/iou_dict to box_p0i_k10_all/iou_dict
"""

import os
import shutil
import glob
from pathlib import Path

def consolidate_box_folders():
    # Base directory
    base_dir = "/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/l2d_models"
    
    # Target consolidated folder
    consolidated_dir = os.path.join(base_dir, "box_p0i_k10_all")
    data_pkl_dir = os.path.join(consolidated_dir, "data_pkl")
    iou_dict_dir = os.path.join(consolidated_dir, "iou_dict")
    
    print(f"Base directory: {base_dir}")
    print(f"Consolidated directory: {consolidated_dir}")
    
    # Create the consolidated folder and subfolders
    print("\nCreating consolidated folder structure...")
    os.makedirs(data_pkl_dir, exist_ok=True)
    os.makedirs(iou_dict_dir, exist_ok=True)
    print(f"✓ Created {consolidated_dir}")
    print(f"✓ Created {data_pkl_dir}")
    print(f"✓ Created {iou_dict_dir}")
    
    # Find all box_p0i_k10_* folders
    pattern = os.path.join(base_dir, "box_p0i_k10_*")
    box_folders = glob.glob(pattern)
    
    # Filter out the consolidated folder if it exists
    box_folders = [f for f in box_folders if not f.endswith("box_p0i_k10_all")]
    
    print(f"\nFound {len(box_folders)} box_p0i_k10_* folders to process:")
    for folder in sorted(box_folders):
        print(f"  - {os.path.basename(folder)}")
    
    # Process each box folder
    total_files_copied = 0
    
    for box_folder in sorted(box_folders):
        folder_name = os.path.basename(box_folder)
        print(f"\nProcessing {folder_name}...")
        
        # Process data_pkl folder
        source_data_pkl = os.path.join(box_folder, "data_pkl")
        if os.path.exists(source_data_pkl):
            files_in_data_pkl = os.listdir(source_data_pkl)
            print(f"  Found {len(files_in_data_pkl)} files in data_pkl")
            
            for file in files_in_data_pkl:
                source_file = os.path.join(source_data_pkl, file)
                dest_file = os.path.join(data_pkl_dir, file)
                
                # Handle filename conflicts by adding folder prefix
                if os.path.exists(dest_file):
                    name, ext = os.path.splitext(file)
                    new_name = f"{folder_name}_{name}{ext}"
                    dest_file = os.path.join(data_pkl_dir, new_name)
                
                shutil.copy2(source_file, dest_file)
                total_files_copied += 1
        else:
            print(f"  ⚠️  data_pkl folder not found in {folder_name}")
        
        # Process iou_dict folder
        source_iou_dict = os.path.join(box_folder, "iou_dict")
        if os.path.exists(source_iou_dict):
            files_in_iou_dict = os.listdir(source_iou_dict)
            print(f"  Found {len(files_in_iou_dict)} files in iou_dict")
            
            for file in files_in_iou_dict:
                source_file = os.path.join(source_iou_dict, file)
                dest_file = os.path.join(iou_dict_dir, file)
                
                # Handle filename conflicts by adding folder prefix
                if os.path.exists(dest_file):
                    name, ext = os.path.splitext(file)
                    new_name = f"{folder_name}_{name}{ext}"
                    dest_file = os.path.join(iou_dict_dir, new_name)
                
                shutil.copy2(source_file, dest_file)
                total_files_copied += 1
        else:
            print(f"  ⚠️  iou_dict folder not found in {folder_name}")
    
    # Summary
    print(f"\n{'='*60}")
    print("CONSOLIDATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total files copied: {total_files_copied}")
    print(f"Consolidated folder: {consolidated_dir}")
    
    # Show final structure
    print(f"\nFinal structure:")
    print(f"  {consolidated_dir}/")
    print(f"  ├── data_pkl/ ({len(os.listdir(data_pkl_dir))} files)")
    print(f"  └── iou_dict/ ({len(os.listdir(iou_dict_dir))} files)")

if __name__ == "__main__":
    try:
        consolidate_box_folders()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
