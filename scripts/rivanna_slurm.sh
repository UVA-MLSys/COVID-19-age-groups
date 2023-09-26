#!/usr/bin/env bash
#SBATCH --job-name="train"
#SBATCH --output=train.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc
cd ..
module load cuda cudnn

# 1. using singularity
module load singularity
python run.py --use_gpu --data_path Top_20.csv --model DLinear

# # 2. using anaconda
#  module load anaconda
#  conda deactivate 
#  conda activate ml # replace with your own virtual env
#  python run.py --use_gpu --data_path Top_20.csv --model DLinear

# # 3. just using local env
#  python run.py --use_gpu --data_path Top_20.csv --model DLinear