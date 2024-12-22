#!/usr/bin/env bash
#SBATCH --job-name="OFA"
#SBATCH --output=scripts/outputs/OFA_LLAMA.out
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load miniforge
# conda deactivate
conda activate ml

# python run_OFA.py --data_path Total.csv \
#     --learning_rate 0.0005 \
#     --d_model 768 \
#     --gpt_layers 2 \
#     --dropout 0.3 --percent 10

python run_OFA.py --data_path Total.csv \
    --learning_rate 0.0005 \
    --d_model 4096 \
    --gpt_layers 2 \
    --dropout 0.3 --llm_model LLAMA\
    --result_path scratch --percent 10