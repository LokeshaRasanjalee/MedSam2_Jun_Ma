#!/bin/bash
#SBATCH --job-name=job_4
#SBATCH -p a100
#SBATCH -n 1                       # number of cores
#SBATCH --time=4:00:00          # time allocation (D-HH:MM:SS)
#SBATCH --gres=gpu:1,tmpfs:50G     # request 1 GPU + tmpfs
#SBATCH --mem=30GB                 # memory per node
#SBATCH --signal B:USR2

# Notification configuration 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lokesha.thennakoonmudiyanselage@adelaide.edu.au

# --- Array configuration ---
# Example: run tasks with index 0–9
#SBATCH --array=0-49

# --- Setup environment ---
source /gpfs/apps/icl/software/Anaconda3/2023.03/etc/profile.d/conda.sh
conda activate medsam2

# --- Go to working directory ---
cd /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma

# --- Define index variable ---
INDEX=${SLURM_ARRAY_TASK_ID}

# --- Run Python with index-specific arguments ---
python L2D_batch_train.py \
 --base_video_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/Barretts_data/dataset2_100_1_2/Images/ \
 --input_mask_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/Barretts_data/dataset2_100_1_2/Masks/ \
 --post_hoc_model_save_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/l2d_models \
 --output_mask_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/media/Inference \
 --array_id ${INDEX} \
 --num_chunks 50 \
 --experiment_name "point_3_p0i_k10_${INDEX}" \
 --sam2_cfg configs/sam2.1_hiera_t512.yaml \
 --sam2_checkpoint /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/checkpoints/sam2.1_hiera_tiny.pt \
 --sample_factor 10 \
 --prompt 'point' 'point' \ 
 --num_clicks 3 

conda deactivate
    
