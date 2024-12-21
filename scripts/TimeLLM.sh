# #!/usr/bin/env bash
# #SBATCH --job-name="TimeLLM"
# #SBATCH --output=scripts/outputs/TimeLLM.out
# #SBATCH --partition=gpu
# #SBATCH --time=1:00:00
# #SBATCH --gres=gpu:1
# #---SBATCH --nodelist=lynx01
# #SBATCH --mem=32GB

# source /etc/profile.d/modules.sh
# source ~/.bashrc

# module load miniforge

# conda activate ml

python3 run_TimeLLM.py --data_path Top_20.csv \
    --model TimeLLM --d_model 4096 \
    --learning_rate 0.0005 --llm_model LLAMA \
    --llm_layers 2 --batch_size 16
