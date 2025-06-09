import os
import pickle
import pandas as pd
import glob
import torch

# Path to the directory containing pickle files
pkl_dir = "/hpcfs/users/a1917962/Jun_Ma_MedSAM2/MedSam2_Jun_Ma/l2d_models/Barrets_dataset_len-10_frameinterval-2_interclipstride-1-all/data_pkl"

# List to store all data
data = []

# Get all pickle files in the directory
pkl_files = glob.glob(os.path.join(pkl_dir, "*.pkl"))
print (pkl_files)

# Read each pickle file
for pkl_file in pkl_files:
    try:
        with open(pkl_file, 'rb') as f:
            pkl_data = pickle.load(f)
            
            # Extract video name from the pickle file name
            video_name = os.path.basename(pkl_file).replace('.pkl', '')
            
            # Extract the required values and convert tensors to float values
            L_no_defer_sam_loss = float(pkl_data['L_no_defer_sam_loss'].item() if torch.is_tensor(pkl_data['L_no_defer_sam_loss']) else pkl_data['L_no_defer_sam_loss'])
            L_post_defer_sam_loss_list = [float(x.item() if torch.is_tensor(x) else x) for x in pkl_data['L_post_defer_sam_loss_list']]
            L_no_defer = float(pkl_data['L_no_defer'].item() if torch.is_tensor(pkl_data['L_no_defer']) else pkl_data['L_no_defer'])
            L_post_defer_list = [float(x.item() if torch.is_tensor(x) else x) for x in pkl_data['L_post_defer_list']]
            
            # Create a row with all values
            row = [video_name, L_no_defer_sam_loss] + L_post_defer_sam_loss_list + [L_no_defer] + L_post_defer_list
            data.append(row)
            
    except Exception as e:
        print(f"Error reading {pkl_file}: {str(e)}")

# Create DataFrame
df = pd.DataFrame(data, columns=['video_name', 'L_no_defer_sam_loss', 'post_defer_1', 'post_defer_2', 'post_defer_3', 'post_defer_4',
                                'post_defer_5', 'post_defer_6', 'post_defer_7', 'post_defer_8','post_defer_9',
                                'L_no_defer', 'L_post_defer_1', 'L_post_defer_2', 'L_post_defer_3', 'L_post_defer_4',
                                'L_post_defer_5', 'L_post_defer_6', 'L_post_defer_7', 'L_post_defer_8', 'L_post_defer_9'])

# Create separate DataFrame for SAM losses
# sam_loss_columns = ['L_no_defer_sam_loss', 'post_defer_1', 'post_defer_2', 'post_defer_3', 'post_defer_4','post_defer_5', 'post_defer_6', 'post_defer_7', 'post_defer_8','post_defer_9']
# df_sam = df[['video_name'] + sam_loss_columns]

# Save to CSV files
df.to_csv('sam_losses_Barrets_dataset_len-10_frameinterval-2_interclipstride-1-all.csv', index=False)
# df_sam.to_csv('sam_losses_only.csv', index=False)

print(f"Full CSV file saved to: sam_losses.csv")
# print(f"SAM losses CSV file saved to: sam_losses_only.csv")
print(f"Total rows processed: {len(data)}") 