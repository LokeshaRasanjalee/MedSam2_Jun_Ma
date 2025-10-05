#!/bin/bash
#SBATCH --job-name=job
#SBATCH -p batch
#SBATCH -n 1                       # number of cores
#SBATCH --time=01:00:00          # time allocation (D-HH:MM:SS)
#SBATCH --mem=10GB                 # memory per node
#SBATCH --signal B:USR2

# Notification configuration 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lokesha.thennakoonmudiyanselage@adelaide.edu.au

# --- Array configuration ---
# Example: run tasks with index 0–9
#SBATCH --array=0-9

# --- Setup environment ---
source /gpfs/apps/icl/software/Anaconda3/2023.03/etc/profile.d/conda.sh
conda activate medsam2

# --- Go to working directory ---
cd /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma

# --- Define index variable ---
INDEX=${SLURM_ARRAY_TASK_ID}

python L2D_train.py \
 --base_video_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/Barretts_data/dataset2_100_1_2/Images/ \
 --input_mask_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/Barretts_data/dataset2_100_1_2/Masks/ \
 -o ./media/Inference \
 --num_groups 10 \
 --array_id ${INDEX} \
 --loss_type iou \
 --data_pkl_dir  /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/l2d_models/mask_p0i_k10_all/data_pkl/ \
 --sample_factor 10 \
 --experiment_name "mask_data_creation"

conda deactivate
    
