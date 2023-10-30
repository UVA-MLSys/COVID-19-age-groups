python run.py --data_path Top_20.csv --model DLinear
python run.py --data_path Top_20.csv --model TimesNet 
python run.py --data_path Top_20.csv --model DLinear --test
python run.py --data_path Top_100.csv --model DLinear 
python run.py --result_path scratch --data_path Top_100.csv --model Autoformer
python run.py --data_path Top_100.csv --model Fedformer
python run_tft.py --data_path Top_20.csv --result_path scratch --disable_progress

python interpret.py --data_path Top_20.csv --model FEDformer --explainer augmented_occlusion --flag test --result_path scratch
python interpret.py --data_path Total.csv --model FEDformer --explainer augmented_occlusion --flag test

python interpret_black_box_metrics.py \
    --data_path Top_100.csv --model FEDformer \
    --explainers feature_ablation --flag test \
    --result_path scratch