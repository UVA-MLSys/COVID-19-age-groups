# local classes and methods
from experiment.tft import Experiment_TFT
from experiment.config import Split, DataConfig, FeatureFiles
from utils.utils import *
from utils.interpreter import *
from explainers import *
from utils.plotter import PlotResults

import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch, os
from datetime import datetime
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer
    

def explainer_factory(
    args, model, dataloader:AgeDataLoader, 
)-> BaseExplainer:
    # only interpreting static reals for now
    features = dataloader.static_reals
    
    if args.explainer == 'FO':
        explainer = FeatureOcclusion(model, dataloader, features)
    elif args.explainer == 'AFO':
        explainer = AugmentedFeatureOcclusion(model, dataloader, features, n_samples=2)
    elif args.explainer == 'FA':
        explainer = FeatureAblation(model, dataloader, features, method='global')
    else:
        raise ValueError(f'{args.explainer} isn\'t supported.')
    return explainer

def main(args):
    # --------- Initialization ---------
    start = datetime.now()
    print(f'Experiment started at {start}.\nCuda available: {torch.cuda.is_available()}.\n')
    
    # Setting random seed
    seed_torch(args.seed)
    
    print(f'Starting experiment. Result folder {args.result_folder}.')
    # clear_directory(args.result_folder)
    
    # get experiment config 
    data_path = os.path.join(args.input_folder, args.input)
    experiment = Experiment_TFT(
        data_path, args.result_folder, not args.disable_progress
    )
    
    # ----- Data Preprocessing and Model Loading ------
    # load dataset
    dataloader = experiment.age_dataloader
    total_data = dataloader.read_df()
    print(total_data.shape)
    print(total_data.head(3))
    
    # split data into train, validation, test
    train_data, _, _ = experiment.age_dataloader.split_data(
        total_data, Split.primary()
    )
    
    # load the best checkpointed model
    best_model_path = get_best_model_path(args.result_folder)
    print(f'Loading best model from {best_model_path}.\n\n')
    model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # --------- Interpret Predictions ---------
    # initialize explainer
    explainer = explainer_factory(args, model, dataloader)
    
    # train any baseline or parameters
    explainer.train_generators(train_data)
    all_scores = explainer.attribute(train_data, args.disable_progress)
    
    # save raw scores
    score_file = os.path.join(args.result_folder, 'scores.npy')
    np.savez_compressed(score_file, all_scores)

    time_index = dataloader.time_index
    features = dataloader.static_reals # only interpreting static features

    # filter out days outside interpretation range
    time_range = explainer.time_range(train_data)
    df = train_data[
        (train_data[time_index]>=time_range[0]) & 
        (train_data[time_index]<=time_range[-1])
    ][['Date', 'FIPS']]
    
    global_rank = calculate_global_rank(
        df, all_scores, features
    )
    global_rank.to_csv(
        os.path.join(args.result_folder, 'global_rank.csv'), index=False
    )
    
    group_agg_scores_df = align_interpretation(df, all_scores, features)
    group_agg_scores_df.to_csv(
        os.path.join(args.result_folder, 'group_agg_scores.csv'), index=False
    )
    
    # plot local interpretations
    plotter = PlotResults(
        figPath=args.result_folder, targets=dataloader.targets, 
        show=not args.disable_progress
    )
    plotter.local_interpretation(
        group_agg_scores_df, features
    )
    
    # ----- Evaluate ------
    # Load ground truth
    group_cases = pd.read_csv(
        os.path.join(FeatureFiles.root_folder, 'Cases by age groups.csv')
    )
    group_cases['end_of_week'] = pd.to_datetime(group_cases['end_of_week'])
    
    # find a common start point
    first_common_date = find_first_common_date(
        group_cases, group_agg_scores_df['Date'].values
    )
    
    # since age group ground truth is weekly aggregated
    # do the same for predicted importance
    weekly_agg_scores_df = aggregate_importance_by_window(
        group_agg_scores_df, dataloader.static_reals, first_common_date
    )
    evaluate_interpretation(
        group_cases, weekly_agg_scores_df, dataloader.static_reals
    )
    

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
        '--seed', type=int, default=7,
        help='seed for randomization'
    )
    
    parser.add_argument(
        '--explainer', type=str, default='FO',
        choices=['FO', 'AFO', 'FA'],
        help="Interpretation method"
    )
    
    return parser

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)