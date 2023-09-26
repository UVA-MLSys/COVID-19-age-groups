#!/usr/bin/env bash
#SBATCH --job-name="train"
#SBATCH --output=scripts/outputs/train.out
#SBATCH --partition=gpu
#SBATCH --time=0:30:00
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn

# 1. using singularity
module load singularity
# outputing to scrach folder ensures temporary experiments aren't tracked
# since scratch is added in gitignore
singularity run --nv timeseries.sif python run.py --use_gpu --result_path scratch --data_path Top_20.csv --model DLinear

# # 2. using anaconda
#  module load anaconda
#  conda deactivate 
#  conda activate ml # replace with your own virtual env
#  python run.py --use_gpu --result_path scratch --data_path Top_20.csv --model DLinear

# # 3. just using local env
#  python run.py --use_gpu --result_path scratch --data_path Top_20.csv --model DLinear