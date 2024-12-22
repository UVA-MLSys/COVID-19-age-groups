#!/usr/bin/env bash
#SBATCH --job-name="TimesNet"
#SBATCH --output=scripts/outputs/TimesNet_train_total.out
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

python3 run.py --data_path Total.csv --model TimesNet --disable_progress

# python3 run.py --data_path Total.csv \
#     --model TimeMixer \
#     --down_sampling_layers 1 \
#     --down_sampling_method avg \
#     --down_sampling_window 2 --disable_progress