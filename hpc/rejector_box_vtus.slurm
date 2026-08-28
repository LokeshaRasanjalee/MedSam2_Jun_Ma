#!/bin/bash
#SBATCH --job-name=n_001
#SBATCH -p a100
#SBATCH -n 1                       # number of cores
#SBATCH --time=04:00:00             # time allocation (D-HH:MM:SS)
#SBATCH --gres=gpu:1,tmpfs:50G      # request 1 GPU + tmpfs
#SBATCH --mem=30GB                  # memory per node
#SBATCH --signal B:USR2

# Notification configuration 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lokesha.thennakoonmudiyanselage@adelaide.edu.au

# --- Array configuration ---
#SBATCH --array=0-0

# --- Setup environment ---
source /gpfs/apps/icl/software/Anaconda3/2023.03/etc/profile.d/conda.sh
conda activate medsam2

# --- Go to working directory ---
cd /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma


# --- Define index variable ---
INDEX=${SLURM_ARRAY_TASK_ID}

# --- Run Python with index-specific arguments ---
python L2D_train.py \
 --base_video_dir ./SUN_data/sun_clip_set_inter_clip_stride_8/Images/ \
 --input_mask_dir ./SUN_data/sun_clip_set_inter_clip_stride_8/Masks/ \
 --data_npz_dir ./miccai_data_pkl_vtus/box_14_10_k10_vtus_all/data_npz_4_iou/ \
 -o ./l2d_train_outputs_miccai_vtus_5000/ \
 --experiment_name "box_vtus_exp_b01_1410_${INDEX}" \
 --batch_size 32 \
 --num_epochs 5000 \
 --learning_rate 1e-7 \
 --alpha 1 \
 --beta 0.1 \
 --full_run True \
 --num_workers 4 \
 --dropout 0 \
 --num_classes 9 \
 --topk_values 1 3 \
 --loss_type 'exp' \
 --distance_weight 0 \
 --array_id ${INDEX} \
 --split_dict_path ./datasets/VTUS/VTUS_created_datasets/vtus_len-30_frameinterval-1_interclipstride-4/split_dict_vtus.txt \
 --tensorboard_status True \
 --save_every 10 \
 --save_model True \
 --dataset 'vtus' \
 --seed ${INDEX} \
 --cost_tau 0.25


conda deactivate