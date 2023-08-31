#!/usr/bin/env bash
#SBATCH --job-name="train"
#SBATCH --output=results/train.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu
#SBATCH --mem=32GB


source /etc/profile.d/modules.sh
source ~/.bashrc

# 1. when you are using singularity
module load cuda cudnn singularity

cd ..
# this is for when you are using singularity
singularity run --nv ./tft_pytorch.sif python run.py \
    --input-folder dataset/processed \
    --input Top_100.csv \
    --result-folder results/top_100 \
    --disable-progress \
    --seed = 7