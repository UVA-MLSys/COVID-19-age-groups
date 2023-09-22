# local classes and methods
from exp.tft import Experiment_TFT
from exp.config import Split, DataConfig, ModelConfig
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
    
    print(f'Starting experiment. Result folder {args.result_folder}.')
    # clear_directory(args.result_folder)
    
    data_path = os.path.join(args.input_folder, args.input)
    experiment = Experiment_TFT(
        data_path, args.result_folder, not args.disable_progress
    )
    total_data = experiment.age_dataloader.read_df()
    print(total_data.shape)
    print(total_data.head(3))
    
    train_data, val_data, test_data = experiment.age_dataloader.split_data(
        total_data, Split.primary()
    )
    
    if args.test:
        best_model_path = get_best_model_path(args.result_folder)
        print(f'Loading best model from {best_model_path}.\n\n')
        model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    else:
        model = experiment.train(
            ModelConfig.primary(), train_data, val_data, ckpt_path=None
        )
    
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
    
    output_file_path = os.path.join(args.result_folder, 'predictions.csv')
    df.to_csv(output_file_path, index=False)

    print(df.head(3))
    print(f'\nEnded at {datetime.now()}. Elapsed time {datetime.now() - start}')
    # torch.cuda.empty_cache()

def get_argparser():
    parser = ArgumentParser(
        description='Run infection prediction model',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-folder', type=str, default=DataConfig.root_folder, 
        help='folder containing the input data file'
    )
    parser.add_argument(
        '--input', type=str, default='Top_100.csv',
        help='input file containing all features'
    )
    parser.add_argument(
        '--result-folder', type=str, default='results', 
        help='result output folder'
    )
    parser.add_argument(
        '--disable-progress', action='store_true', 
        help='disable progress bar. useful when submitting job script.'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='test the best model at result folder'
    )
    parser.add_argument(
        '--seed', type=int, default=7,
        help='seed for randomization'
    )
    return parser

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
