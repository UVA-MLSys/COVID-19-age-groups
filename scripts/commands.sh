python run.py --use_gpu --data_path Top_20.csv --model DLinear
python run.py --use_gpu --data_path Top_20.csv --model Transformer 
python run.py --use_gpu --data_path Top_20.csv --model Transformer --test
python run.py --use_gpu --data_path Top_100.csv --model Transformer 
python run.py --result_path scratch --use_gpu --data_path Top_100.csv --model Autoformer
python run.py --use_gpu --data_path Top_100.csv --model Fedformer --n_heads 8