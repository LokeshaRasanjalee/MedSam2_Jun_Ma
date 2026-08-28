#!/bin/bash
#SBATCH --job-name=job
#SBATCH -p batch
#SBATCH -n 1                       # number of cores
#SBATCH --time=00:30:00          # time allocation (D-HH:MM:SS)
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
 --base_video_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/datasets/VTUS/VTUS_created_datasets/vtus_len-30_frameinterval-1_interclipstride-4/Images/ \
 --input_mask_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/datasets/VTUS/VTUS_created_datasets/vtus_len-30_frameinterval-1_interclipstride-4/Masks/ \
 -o ./media/Inference \
 --num_groups 10 \
 --array_id ${INDEX} \
 --loss_type iou \
 --data_pkl_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/miccai_data_pkl_vtus_layer4/box_14_18_k10_vtus_all/data_pkl/ \
 --sample_factor 10 \
 --experiment_name "box_14_18_k10_vtus_${INDEX}"

conda deactivate
    
