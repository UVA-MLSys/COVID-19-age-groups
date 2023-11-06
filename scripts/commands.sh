python run.py --data_path Top_20.csv --model DLinear
python run.py --data_path Top_20.csv --model TimesNet 
python run.py --data_path Top_20.csv --model DLinear --test
python run.py --data_path Top_100.csv --model DLinear 
python run.py --result_path scratch --data_path Top_100.csv --model Autoformer
python run.py --data_path Top_100.csv --model FEDformer

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