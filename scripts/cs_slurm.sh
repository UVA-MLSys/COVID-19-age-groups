#!/usr/bin/env bash
#SBATCH --job-name="train_total"
#SBATCH --output=scripts/outputs/train_total.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn anaconda3
conda deactivate
conda activate ml

python3 run.py --use_gpu --result_path scratch --data_path Total.csv --model DLinear --num_workers -1