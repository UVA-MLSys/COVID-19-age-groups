#!/usr/bin/env bash
#SBATCH --job-name="CALF"
#SBATCH --output=scripts/outputs/CALF.out
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load miniforge
# conda deactivate
conda activate ml

python3 run_CALF.py --data_path Total.csv \
    --model CALF --d_model 768 \
    --learning_rate 0.0005 \
    --gpt_layers 2 --disable_progress --percent 10
