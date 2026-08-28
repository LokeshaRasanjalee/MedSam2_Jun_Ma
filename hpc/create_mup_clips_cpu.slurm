#!/bin/bash
#SBATCH --job-name=mup_clips
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=8GB
#SBATCH --array=0-5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lokesha.thennakoonmudiyanselage@adelaide.edu.au


source /gpfs/apps/icl/software/Anaconda3/2023.03/etc/profile.d/conda.sh
conda activate medsam2

cd /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma

python CMIG_data_processing/create_mup_clips.py \
    --batch-index "${SLURM_ARRAY_TASK_ID}" \
    --batch-size 10 \
    --overwrite

conda deactivate
