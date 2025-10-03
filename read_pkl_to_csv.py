#!/usr/bin/env python3
"""
Script to read pickle files from a directory and convert their contents to CSV format.
Each pickle file contains a dictionary with IoU scores for different thresholds.
"""

import os
import pickle
import pandas as pd
import glob

def read_pkl_files_to_csv(pkl_directory, output_csv_path, threshold_to_keep=50):
    """
    Read all pickle files from a directory and convert to CSV format.
    Saves only the records where threshold == threshold_to_keep.
    
    Args:
        pkl_directory (str): Path to directory containing pickle files
        output_csv_path (str): Path for output CSV file
        threshold_to_keep (int): Threshold value to keep
    """
    
    # Get all pickle files in the directory
    pkl_files = glob.glob(os.path.join(pkl_directory, "*.pkl"))
    
    if not pkl_files:
        print(f"No pickle files found in {pkl_directory}")
        return
    
    print(f"Found {len(pkl_files)} pickle files")
    
    # List to store all data
    all_data = []
    
    # Process each pickle file
    for pkl_file in sorted(pkl_files):
        try:
            # Extract filename without extension for identification
            filename = os.path.basename(pkl_file)
            file_id = filename.replace('_iou_dict.pkl', '')
            
            # Load pickle file
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Only process rows where threshold == threshold_to_keep
            if threshold_to_keep in data:
                iou_scores = data[threshold_to_keep]
                for idx, iou_score in enumerate(iou_scores):
                    row = {
                        'file_id': file_id,
                        'threshold': threshold_to_keep,
                        'score_index': idx,
                        'iou_score': iou_score
                    }
                    all_data.append(row)
            
            print(f"Processed: {filename}")
            
        except Exception as e:
            print(f"Error processing {pkl_file}: {str(e)}")
            continue
    
    # Create DataFrame and save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        print(f"\nData saved to: {output_csv_path}")
        print(f"Total records: {len(df)}")
        print(f"Unique files: {df['file_id'].nunique()}")
        print(f"Thresholds in data: {df['threshold'].unique()}")
        
        # Display first few rows
        print("\nFirst 10 rows:")
        print(df.head(10))
        
        # Display summary statistics
        print(f"\nSummary statistics for threshold {threshold_to_keep}:")
        print(df['iou_score'].describe())
        
    else:
        print(f"No data found for threshold {threshold_to_keep}")

def main():
    # Define paths
    pkl_directory = "/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/l2d_models/mask_p0i_k10_all/iou_dict"
    output_csv_path = "/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/iou_scores_data_threshold_0_mask.csv"
    threshold_to_keep = '0'   # <-- only keep threshold=50
    
    # Check if directory exists
    if not os.path.exists(pkl_directory):
        print(f"Directory does not exist: {pkl_directory}")
        return
    
    print(f"Reading pickle files from: {pkl_directory}")
    print(f"Output CSV will be saved to: {output_csv_path}")
    
    # Process files
    read_pkl_files_to_csv(pkl_directory, output_csv_path, threshold_to_keep)

if __name__ == "__main__":
    main()
