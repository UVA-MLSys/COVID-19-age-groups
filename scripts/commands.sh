python run.py --data_path Top_20.csv --model DLinear
python run.py --data_path Top_20.csv --model TimesNet 
python run.py --data_path Top_20.csv --model DLinear --test
python run.py --data_path Top_100.csv --model DLinear 
python run.py --result_path scratch --data_path Top_100.csv --model Autoformer
python run.py --data_path Top_100.csv --model FEDformer

python run_CALF.py --data_path Top_20.csv \
    --model CALF \
    --gpt_layers 2 \
    --d_model 768

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

python run_OFA.py --data_path Top_20.csv \
    --model OFA \
    --gpt_layers 2 \
    --is_gpt 1 \
    --patch_size 7 \
    --kernel_size 8 \
    --pretrain 1 \
    --freeze 1 \
    --stride 7 \
    --max_len -1 \
    --hid_dim 16 \
    --tmax 10 \
    --n_scale -1 \
    --batch_size 16