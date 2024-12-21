python run.py --data_path Top_20.csv --model DLinear
python run.py --data_path Top_20.csv --model TimesNet 
python run.py --data_path Top_20.csv --model DLinear --test
python run.py --data_path Top_100.csv --model DLinear 
python run.py --result_path scratch --data_path Top_100.csv --model Autoformer
python run.py --data_path Top_100.csv --model FEDformer

python run_CALF.py --data_path Top_20.csv --model CALF --d_model 768 --dropout 0.3 --learning_rate 0.0005 --gpt_layers 2

python run_TimeLLM.py --data_path Top_20.csv --model CALF --llm_dim 768 --d_model 768 --batch_size 16

python interpret_with_ground_truth.py --data_path Top_20.csv \
    --model FEDformer \
    --explainers feature_ablation augmented_occlusion \
    --flag test --result_path scratch

python interpret_with_ground_truth.py --data_path Total.csv \
    --model FEDformer --flag test \
    --explainers augmented_occlusion 

python interpret_without_ground_truth.py \
    --data_path Top_20.csv --model FEDformer \
    --explainers feature_ablation --flag test \
    --result_path scratch

python run_OFA.py --data_path Top_20.csv --learning_rate 0.0005 --d_model 768 --gpt_layer 6 --dropout 0.3 

python run_OFA.py --data_path Top_20.csv \
    --learning_rate 0.0001 \
    --model OFA \
    --patch_size 7 \
    --stride 1 \
    --batch_size 32 --d_model 768 \
    --gpt_layer 6 --dropout 0.3 