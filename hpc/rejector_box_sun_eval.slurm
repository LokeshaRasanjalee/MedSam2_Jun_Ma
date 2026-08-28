#!/bin/bash
#SBATCH --job-name=k_001
#SBATCH -p a100
#SBATCH -n 1                       # number of cores
#SBATCH --time=01:00:00             # time allocation (D-HH:MM:SS)
#SBATCH --gres=gpu:1,tmpfs:50G      # request 1 GPU + tmpfs
#SBATCH --mem=30GB                  # memory per node
#SBATCH --signal B:USR2

# Notification configuration 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lokesha.thennakoonmudiyanselage@adelaide.edu.au

# --- Array configuration ---
#SBATCH --array=1-1

# --- Setup environment ---
source /gpfs/apps/icl/software/Anaconda3/2023.03/etc/profile.d/conda.sh
conda activate medsam2

# --- Go to working directory ---
cd /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma


# --- Define index variable ---
INDEX=${SLURM_ARRAY_TASK_ID}

# --- Run Python with index-specific arguments ---
python L2D_eval.py \
 --save_csv_path ./l2d_train_outputs_miccai_sun/20260203_235540_box_sun_mae_b0_1410_1/eval_logits.csv \
 --split_dict_path ./datasets/SUN_data/sun_clip_set_inter_clip_stride_8/split_dict_sun.txt \
 --dataset 'sun' \
 --seed 42 \
 --cost_tau 0.25 \
 --num_classes 9 \
 --dropout 0 \
 --batch_size 32 \
 --data_npz_dir ./miccai_data_pkl_sun/box_14_10_k10_sun_all/data_npz_4_iou/ \
 --eval_model_path ./l2d_train_outputs_miccai_sun/20260203_235540_box_sun_mae_b0_1410_1/model_box_sun_mae_b0_1410_1_last_epoch_1000_ma_acc_0.5020.pth

conda deactivate