#!/bin/bash
#SBATCH --job-name=job
#SBATCH -p a100
#SBATCH -n 1                       # number of cores
#SBATCH --time=1:30:00          # time allocation (D-HH:MM:SS)
#SBATCH --gres=gpu:1,tmpfs:50G     # request 1 GPU + tmpfs
#SBATCH --mem=30GB                 # memory per node
#SBATCH --signal B:USR2

# Notification configuration 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lokesha.thennakoonmudiyanselage@adelaide.edu.au

# --- Array configuration ---
# Example: run tasks with index 0–9
#SBATCH --array=0-3

# --- Setup environment ---
source /gpfs/apps/icl/software/Anaconda3/2023.03/etc/profile.d/conda.sh
conda activate medsam2

# --- Go to working directory ---
cd /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma

# --- Define index variable ---
INDEX=${SLURM_ARRAY_TASK_ID}

# --- Run Python with index-specific arguments ---
python L2D_batch_train.py \
 --base_video_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/datasets/VTUS/VTUS_created_datasets/vtus_len-30_frameinterval-1_interclipstride-4/Images/ \
 --input_mask_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/datasets/VTUS/VTUS_created_datasets/vtus_len-30_frameinterval-1_interclipstride-4/Masks/ \
 --post_hoc_model_save_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/miccai_data_pkl_vtus/ \
 --output_mask_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/media/Inference \
 --array_id ${INDEX} \
 --num_chunks 4 \
 --experiment_name "box_14_12_k10_vtus_${INDEX}" \
 --sam2_cfg configs/sam2.1_hiera_t512.yaml \
 --sam2_checkpoint /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/checkpoints/sam2.1_hiera_tiny.pt \
 --sample_factor 10 \
 --prompt 'box' 'box' \
 --human_prompt 1.2 \
 --machine_prompt 1.4

conda deactivate
    
