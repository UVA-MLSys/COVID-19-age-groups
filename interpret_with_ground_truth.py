import os, gc
import pandas as pd
from datetime import datetime
from os.path import join

import warnings
warnings.filterwarnings('ignore')

from captum.attr import Lime, DeepLift, IntegratedGradients, GradientShap
from explainers import MorrisSensitivty
from tint.attr import (
    AugmentedOcclusion,
    Occlusion, 
    FeatureAblation
)

from run import stringify_setting, get_parser as get_run_parser, initial_setup
from exp.config import FeatureFiles, DataConfig
from exp.exp_forecasting import Exp_Forecast
from exp.exp_interpret import initialize_explainer, explainer_name_map
from interpret_with_ground_truth import *
from utils.interpreter import *

explainer_map = {
    'feature_ablation': FeatureAblation,
    'occlusion': Occlusion,
    'augmented_occlusion': AugmentedOcclusion,
    'lime': Lime,
    'deep_lift': DeepLift,
    'integrated_gradients': IntegratedGradients,
    'gradient_shap': GradientShap,
    'morris_sensitivity': MorrisSensitivty
}

def main(args):
    print(f'Experiment started at {datetime.now()}')
    # only has real features and observed reals also contains past targets
    features = DataConfig.static_reals + DataConfig.observed_reals
    age_features = DataConfig.static_reals
    
    # update arguments
    initial_setup(args)
    
    setting = stringify_setting(args)
    
    # initialize experiment
    exp = Exp_Forecast(args, setting)  # set experiments
    exp.load_model()
    
    # get dataset and dataloader
    flag = args.flag
    dataset, dataloader = exp.get_data(flag)
    
    # get ground truth
    df = exp.data_map[flag]
    df.sort_values(by=['Date', 'FIPS'], inplace=True)
    
    # read ground truth and county populations
    group_cases = pd.read_csv(
        join(FeatureFiles.root_folder, 'Cases by age groups.csv')
    )
    group_cases['end_of_week'] = pd.to_datetime(group_cases['end_of_week'])

    population = pd.read_csv(join(FeatureFiles.root_folder, 'Population.csv'))
    population = population[['FIPS', 'POPESTIMATE']]
    # weight attributions by population ratio and total count
    weights = df.groupby('FIPS').first()[age_features].reset_index()
    
    # create result folder if not present
    result_folder = os.path.join(exp.output_folder, 'interpretation')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    print(f'Interpretation results will be saved in {result_folder}')
    
    for explainer_name in args.explainers:
        # calculate attribute
        start = datetime.now()
        print(f'{explainer_name} interpretation started at {start}')
        explainer = initialize_explainer(
            explainer_name, exp, dataloader, args, add_x_mark=False
        )
        
        # batch x pred_len x seq_len x features
        attr = batch_compute_attr(
            dataloader, exp, explainer, 
            baseline_mode=args.baseline_mode,
            add_x_mark=False # only interpret the static and dynamic features
        )
        
        # batch x pred_len x seq_len x features -> batch x pred_len x features
        attr = attr.mean(axis=2)
        # batch x features x pred_len
        attr = attr.permute(0, 2, 1)
        
        end = datetime.now()
        print(f'{explainer_name} interpretation ended at {end}, total time {end - start}')

        # taking absolute since we want the magnitude of feature importance only
        attr_numpy = np.abs(attr.detach().cpu().numpy())

        # align attribution to date time index
        attr_df = align_interpretation(
            ranges=dataset.ranges,
            attr=attr_numpy,
            features=features,
            min_date=df['Date'].min(),
            seq_len=args.seq_len, pred_len=args.pred_len
        )
        print('Attribution statistics')
        print(attr_df.describe())
        gc.collect()
        
        # multiply the importance of age groups from each county by the corresponding population
        groups = []
        for FIPS, group_df in attr_df.groupby('FIPS'):
            county_age_weights = weights[weights['FIPS']==FIPS][age_features].values
            total_population = population[
                population['FIPS']==FIPS]['POPESTIMATE'].values[0]
            group_df[age_features] *= county_age_weights * total_population
            groups.append(group_df)
            
        groups = pd.concat(groups, axis=0)
        weighted_attr_df = groups[['FIPS', 'Date'] + age_features].reset_index(drop=True)

        weighted_attr_by_date = weighted_attr_df.groupby('Date')[
            age_features].aggregate('sum').reset_index()
        
        dates = weighted_attr_by_date['Date'].values
        first_common_date = find_first_common_date(group_cases, dates)
        last_common_date = find_last_common_date(group_cases, dates)

        # sum of ground truth cases within common time
        summed_ground_truth = group_cases[
            (group_cases['end_of_week']>=first_common_date) &
            (group_cases['end_of_week']<=last_common_date)
        ][age_features].mean(axis=0).T.reset_index()
        summed_ground_truth.columns = ['age_group', 'cases']
        
        # sum of predicted weighted age relevance score within common time
        summed_weighted_attr = weighted_attr_df[
            (weighted_attr_df['Date']>=(first_common_date-pd.to_timedelta(6, unit='D'))) &
            (weighted_attr_df['Date']<=last_common_date)
        ][age_features].mean(axis=0).T.reset_index()
        summed_weighted_attr.columns = ['age_group', 'attr']
        
        # merge ground truth and predicted ranking
        global_rank = summed_ground_truth.merge(
            summed_weighted_attr, on='age_group', how='inner'
        ) 
        global_rank[['cases', 'attr']] = global_rank[['cases', 'attr']].div(
            global_rank[['cases', 'attr']].sum(axis=0)/100, axis=1
        ).fillna(0) # will be null when all attributions are zero

        global_rank['cases_rank'] = global_rank['cases'].rank(
            axis=0, ascending=False
        )
        global_rank['attr_rank'] = global_rank['attr'].rank(
            axis=0, ascending=False
        )
        print('Global rank comparison')
        print(global_rank)
        global_rank.to_csv(
            join(
                result_folder, 
                f'{flag}_global_rank_{explainer.get_name()}.csv'
            ), 
            index=False
        )
        
        print('\nEvaluating local ranks')
        # since age group ground truth is weekly aggregated
        # do the same for predicted importance
        weekly_agg_scores_df = aggregate_importance_by_window(
            weighted_attr_by_date, age_features, first_common_date
        )
        result_df = evaluate_interpretation(
            group_cases, weekly_agg_scores_df, age_features
        )
        result_df.to_csv(
            join(
                result_folder, 
                f'{flag}_int_metrics_{explainer.get_name()}.csv'
            ), 
            index=False
        )

def get_parser():
    parser = get_run_parser()
    parser.description = 'Interpret Timeseries Models'
    
    parser.add_argument('--explainers', nargs='*', default=['feature_ablation'], 
        choices=list(explainer_name_map.keys()),
        help='explaination method names')
    parser.add_argument('--flag', type=str, default='test', 
        choices=['train', 'val', 'test', 'updated'],
        help='flag for data split'
    )
    parser.add_argument('--baseline_mode', type=str, default='random',
        choices=['random', 'aug', 'zero', 'mean'],
        help='how to create the baselines for the interepretation methods'
    )
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)