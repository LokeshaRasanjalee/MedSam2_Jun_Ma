#!/bin/bash
#SBATCH --job-name=job
#SBATCH -p batch
#SBATCH -n 1                       # number of cores
#SBATCH --time=30:00          # time allocation (D-HH:MM:SS)
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
 --base_video_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/datasets/Micro_Ultrasound_Prostate_Segmentation_Dataset/MUP_created_datasets/mup_len-10_frameinterval-1_interclipstride-1/micro_ultrasound_scans/ \
 --input_mask_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/datasets/Micro_Ultrasound_Prostate_Segmentation_Dataset/MUP_created_datasets/mup_len-10_frameinterval-1_interclipstride-1/expert_annotations/ \
 -o ./media/Inference \
 --num_groups 10 \
 --array_id ${INDEX} \
 --loss_type iou \
 --data_pkl_dir  /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/miccai_data_pkl_mup/mask_k10_mup_all/data_pkl/ \
 --sample_factor 10 \
 --experiment_name "mask_data_creation"

conda deactivate
    
