#!/bin/bash
#SBATCH --job-name=job
#SBATCH -p a100
#SBATCH -n 1                       # number of cores
#SBATCH --time=06:00:00             # time allocation (D-HH:MM:SS)
#SBATCH --gres=gpu:1,tmpfs:50G      # request 1 GPU + tmpfs
#SBATCH --mem=30GB                  # memory per node
#SBATCH --signal B:USR2

# Notification configuration 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lokesha.thennakoonmudiyanselage@adelaide.edu.au

# --- Array configuration ---
#SBATCH --array=0-3

# --- Setup environment ---
source /gpfs/apps/icl/software/Anaconda3/2023.03/etc/profile.d/conda.sh
conda activate medsam2

# --- Go to working directory ---
cd /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma


# --- Define index variable ---
INDEX=${SLURM_ARRAY_TASK_ID}

# --- Run Python with index-specific arguments ---
python L2D_train.py \
 --base_video_dir ./Barretts_data/dataset_tmp/Images/ \
 --input_mask_dir ./Barretts_data/dataset_tmp/Masks/ \
 --data_npz_dir ./l2d_models/box_p0i_k10_all/data_npz_4_iou/ \
 -o ./l2d_train_outputs/ \
 --experiment_name "box_k10_b0045_lr7_${INDEX}" \
 --batch_size 32 \
 --num_epochs 1000 \
 --learning_rate 1e-7 \
 --alpha 1 \
 --beta 0.045 \
 --full_run True \
 --num_workers 4 \
 --dropout 0 \
 --num_classes 9 \
 --topk_values 1 3 \
 --loss_type 'mae' \
 --array_id ${INDEX} \
 --split_dict_path ./l2d_models/box_p0i_k10_all/split_dict.txt \
 --tensorboard_status True \
 --save_every 10 \
 --save_model True

conda deactivate
