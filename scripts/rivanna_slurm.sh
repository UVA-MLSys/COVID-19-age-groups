#!/usr/bin/env bash
#SBATCH --job-name="train"
#SBATCH --output=scripts/outputs/train.out
#SBATCH --partition=gpu
#SBATCH --time=5:00:00  # can be reduced to 1h for Top 20 ot 500 counties
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32GB # reduce to 16GB for Top 20 ot 500 counties

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn

# 1. using singularity
module load singularity
# outputing to scrach folder ensures temporary experiments aren't tracked
# since scratch is added in gitignore
singularity run --nv timeseries.sif python run.py --data_path Top_20.csv --model FEDformer --disable_progress
# singularity run --nv timeseries.sif python interpret_with_ground_truth.py \
#     --data_path Top_20.csv --model FEDformer \
#     --explainers morris_sensitivity --disable_progress

# # 2. using anaconda
#  module load anaconda
#  conda deactivate 
#  conda activate ml # replace with your own virtual env
#  python run.py --data_path Top_20.csv --model DLinear
# python interpret_with_ground_truth.py --data_path Top_20.csv \
#     --model FEDformer \
#     --explainers feature_ablation occlusion augmented_occlusion feature_permutation morris_sensitivity deep_lift gradient_shap integrated_gradients \
#     --disable_progress

# # 3. just using local env
#  python run.py --data_path Top_20.csv --model DLinear