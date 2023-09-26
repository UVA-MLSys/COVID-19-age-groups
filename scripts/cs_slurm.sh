#!/usr/bin/env bash
#SBATCH --job-name="train"
#SBATCH --output=scripts/outputs/train.out
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mem=16GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn anaconda3
conda deactivate
conda activate ml

python3 run.py --use_gpu --data_path Top_20.csv --model DLinear