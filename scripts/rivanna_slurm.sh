#!/usr/bin/env bash
#SBATCH --job-name="Crossformer_train_total"
#SBATCH --output=scripts/outputs/Crossformer_train_total.out
#SBATCH --partition=gpu
#SBATCH --time=12:00:00  # can be reduced to 1h for Top 20 ot 500 counties
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32GB # reduce to 16GB for Top 20 ot 500 counties
# # SBATCH --mail-type=end
# # SBATCH --mail-user=mi3se@virginia.edu


source /etc/profile.d/modules.sh
source ~/.bashrc

# module load cuda cudnn

# 1. using singularity
module load singularity
# outputing to scrach folder ensures temporary experiments aren't tracked
# since scratch is added in gitignore
singularity run --nv timeseries.sif python run.py --data_path Total.csv --model Crossformer
# singularity run --nv timeseries.sif python interpret_without_ground_truth.py \
#     --data_path Total.csv --model FEDformer \
#     --explainers feature_ablation \
#     --disable_progress --flag updated --batch_size 64

# # 2. using anaconda
# module load anaconda
# conda deactivate 
# conda activate ml # replace with your own virtual env name

# # replace the following with your id and env if facing the lib not found error
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mi3se/.conda/envs/ml/lib

#  python run.py --data_path Top_20.csv --model DLinear
# python interpret_with_ground_truth.py --data_path Total.csv \
#     --model FEDformer \
#     --explainers feature_ablation occlusion augmented_occlusion feature_permutation morris_sensitivity deep_lift gradient_shap integrated_gradients \
#     --disable_progress
    
# python interpret_with_ground_truth.py --data_path Total.csv \
#     --model FEDformer \
#     --explainers feature_ablation occlusion augmented_occlusion feature_permutation morris_sensitivity deep_lift gradient_shap integrated_gradients \
#     --disable_progress

# # 3. just using local env
#  python run.py --data_path Top_20.csv --model DLinear