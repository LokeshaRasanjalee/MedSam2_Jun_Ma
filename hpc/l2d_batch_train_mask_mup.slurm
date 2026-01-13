#!/bin/bash
#SBATCH --job-name=job_mup
#SBATCH -p a100
#SBATCH -n 1                       # number of cores
#SBATCH --time=30:00          # time allocation (D-HH:MM:SS)
#SBATCH --gres=gpu:1,tmpfs:50G     # request 1 GPU + tmpfs
#SBATCH --mem=30GB                 # memory per node
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

# --- Run Python with index-specific arguments ---
python L2D_batch_train.py \
 --base_video_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/datasets/Micro_Ultrasound_Prostate_Segmentation_Dataset/MUP_created_datasets/mup_len-10_frameinterval-1_interclipstride-1/micro_ultrasound_scans/ \
 --input_mask_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/datasets/Micro_Ultrasound_Prostate_Segmentation_Dataset/MUP_created_datasets/mup_len-10_frameinterval-1_interclipstride-1/expert_annotations/ \
 --post_hoc_model_save_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/miccai_data_pkl/ \
 --output_mask_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/media/Inference \
 --array_id ${INDEX} \
 --num_chunks 10 \
 --experiment_name "mask_k10_mup_${INDEX}" \
 --sam2_cfg configs/sam2.1_hiera_t512.yaml \
 --sam2_checkpoint /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/checkpoints/sam2.1_hiera_tiny.pt \
 --sample_factor 10 \
 --prompt 'mask' 'mask' \
 --annotator_2_mask_dir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/datasets/Micro_Ultrasound_Prostate_Segmentation_Dataset/MUP_created_datasets/mup_len-10_frameinterval-1_interclipstride-1/non_expert_annotations/ 


conda deactivate
    
