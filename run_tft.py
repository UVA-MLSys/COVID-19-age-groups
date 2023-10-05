# local classes and methods
from exp.exp_tft import Experiment_TFT
from exp.config import Split, DataConfig
from utils.utils import seed_torch, clear_directory, get_best_model_path

import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch, os
from datetime import datetime
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer

def main(args):
    start = datetime.now()
    print(f'Experiment started at {start}.\nCuda available: {torch.cuda.is_available()}.\n')
    
    # Setting random seed
    seed_torch(args.seed)
    
    setting = stringify_setting(args)
    experiment = Experiment_TFT(args, setting)
    if args.clear:
        clear_directory(experiment.output_folder)
        
    total_data = experiment.age_dataloader.read_df()
    print(total_data.shape)
    print(total_data.head(3))
    
    train_data, val_data, test_data = experiment.age_dataloader.split_data(
        total_data, Split.primary()
    )
    
    if args.test:
        best_model_path = get_best_model_path(experiment.output_folder)
        print(f'Loading best model from {best_model_path}.')
        model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    else:
        model = experiment.train(train_data, val_data)
    
    print('\n---Training prediction--\n')
    train_result_merged = experiment.test(model, train_data, split_type='Train')
    
    print(f'\n---Validation results--\n')
    val_result_merged = experiment.test(model, val_data, split_type='Validation')
    
    print(f'\n---Test results--\n')
    test_result_merged = experiment.test(model, test_data, split_type='Test')

    # Dump results
    train_result_merged['split'] = 'train'
    val_result_merged['split'] = 'validation'
    test_result_merged['split'] = 'test'
    
    df = pd.concat([train_result_merged, val_result_merged, test_result_merged])
    df.sort_values(
        by=[experiment.age_dataloader.date_index] + experiment.age_dataloader.group_ids,
        inplace=True
    )
    
    output_file_path = os.path.join(experiment.output_folder, 'predictions.csv')
    df.to_csv(output_file_path, index=False)

    print(df.head(3))
    print(f'\nEnded at {datetime.now()}. Elapsed time {datetime.now() - start}')
    # torch.cuda.empty_cache()

def get_argparser():
    parser = ArgumentParser(
        description='Run TFT prediction model',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    # data loader
    parser.add_argument(
        '--no_scale', action='store_true',
        help='do not scale dataset'
    )
    parser.add_argument(
        '--clear', action='store_true', help='clear output folder'
    )
    parser.add_argument(
        '--root_path', type=str, default=DataConfig.root_folder, 
        help='folder containing the input data file'
    )
    parser.add_argument(
        '--data_path', type=str, default='Top_100.csv',
        help='input feature file name'
    )
    parser.add_argument(
        '--result_path', type=str, default='results', 
        help='result output folder'
    )
    
    # work configuration
    parser.add_argument(
        '--disable_progress', action='store_true', 
        help='disable progress bar'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='test the best model at result folder'
    )
    parser.add_argument(
        '--seed', type=int, default=7,
        help='seed for randomization'
    )
    
    # model configuration
    parser.add_argument('--batch_size', type=int, default=64, 
        help='batch size of train input data')
    parser.add_argument('--seq_len', type=int, default=14, 
        help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=14, 
        help='prediction sequence length')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--hidden_size', type=int, default=16, 
        help='hidden layer size')
    parser.add_argument('--lstm_layers', type=int, default=1, 
        help='lstm layer numbers')
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, 
        help='optimizer learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--clip', type=float, default=1.0, 
        help='The value at which to clip gradients. Passing gradient_clip_val=None disables gradient clipping.')
    
    return parser

def stringify_setting(args):
    setting = f"TFT_{args.data_path.split('.')[0]}"
    return setting

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
