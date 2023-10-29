import numpy as np
import pandas as pd
from typing import List, Union
from sklearn.metrics import dcg_score, ndcg_score, mean_absolute_error, mean_squared_error
from pandas import DataFrame
import torch
from tqdm import tqdm
from exp.exp_forecasting import Exp_Forecast
from data.dataloader import MultiTimeSeries

def batch_compute_attr(
    dataloader:MultiTimeSeries, exp:Exp_Forecast, 
    explainer, baseline_mode:str = "random",
    include_x_mark=False
) -> torch.TensorType:
    """Computes the attribute of this dataloder in batch using the explainer
    and baseline mode.

    Args:
        dataloader (MultiTimeSeries): torch dataloader
        exp (Exp_Forecast): experimenter class
        explainer: explainer instanace from time interpret
        baseline_mode (str, optional): how baselines passed to 
            the explainer attribute method are generated.
            Available options: [zero, random, aug, mean] . Defaults to "random".

    Returns:
        torch.Tensor: Attributes of the input data, shape is the same as input.
    """
    attr_list = []

    progress_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), 
        disable=exp.args.disable_progress
    )
    for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in progress_bar:
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float().to(exp.device)

        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float()
        # outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        if include_x_mark:
            inputs = (batch_x, batch_x_mark)
            additional_forward_args = (dec_inp, batch_y_mark)
        else:
            inputs = batch_x
            additional_forward_args = (batch_x_mark, dec_inp, batch_y_mark)
        
        # baseline must be a scaler or tuple of tensors with same dimension as input
        baselines = get_baseline(inputs, mode=baseline_mode)

        # get attributions
        attr = compute_attr(
            inputs, baselines, explainer, additional_forward_args, exp.args
        )
        attr_list.append(attr)
        
    if include_x_mark:
        # tuple of n_examples x pred_len x seq_len x features
        attr = (
            [torch.vstack([a[i] for a in attr_list])] for i in range(2))
    else:
        # n_examples x pred_len x seq_len x features
        attr = torch.vstack(attr_list)
    return attr

def compute_regressor_attr(
    inputs, baselines, explainer,
    additional_forward_args, args
):
    name = explainer.get_name()
    if name in ['Deep Lift', 'Lime', 'Integrated Gradients', 'Gradient Shap']:
        attr_list = []
        for target in range(args.pred_len):
            score = explainer.attribute(
                inputs=inputs, baselines=baselines, target=target,
                additional_forward_args=additional_forward_args
            )
            attr_list.append(score)
        
        if type(inputs) == tuple:
            attr = []
            for input_index in range(len(inputs)):
                attr_per_input = torch.stack([score[input_index] for score in attr_list])
                # pred_len x batch x seq_len x features -> batch x pred_len x seq_len x features
                attr_per_input = attr_per_input.permute(1, 0, 2, 3)
                attr.append(attr_per_input)
                
            attr = tuple(attr)
        else:
            attr = torch.stack(attr_list)
            # pred_len x batch x seq_len x features -> batch x pred_len x seq_len x features
            attr = attr.permute(1, 0, 2, 3)
        
    elif name in ['Feature Ablation']:
        attr = explainer.attribute(
            inputs=inputs, baselines=baselines,
            additional_forward_args=additional_forward_args
        )
    elif name in ['Morris Sensitivity', 'Feature Permutation']:
        attr = explainer.attribute(
            inputs=inputs,
            additional_forward_args=additional_forward_args
        )
    elif name == 'Occlusion' or name=='Augmented Occlusion':
        if type(inputs) == tuple:
            sliding_window_shapes = tuple([(1,1) for _ in inputs])
        else:
            sliding_window_shapes = (1,1)
            
        if name == 'Occlusion':
            attr = explainer.attribute(
                inputs=inputs,
                baselines=baselines,
                sliding_window_shapes = sliding_window_shapes,
                additional_forward_args=additional_forward_args
            )
        else:
            attr = explainer.attribute(
                inputs=inputs,
                sliding_window_shapes = sliding_window_shapes,
                additional_forward_args=additional_forward_args
            )
    else:
        raise NotImplementedError
        
    attr = reshape_over_output_horizon(attr, inputs, args)
    return attr

def reshape_over_output_horizon(attr, inputs, args):
    if type(inputs) == tuple:
        # tuple of batch x seq_len x features
        attr = tuple([
            attr_.reshape(
                # batch x pred_len x seq_len x features
                (inputs[0].shape[0], -1, args.seq_len, attr_.shape[-1])
            # take mean over the output horizon
            ) for attr_ in attr
        ])
    else:
        # batch x seq_len x features
        attr = attr.reshape(
            # batch x pred_len x seq_len x features
            (inputs.shape[0], -1, args.seq_len, attr.shape[-1])
        # take mean over the output horizon
        )
    
    return attr

def compute_attr(
    inputs, baselines, explainer,
    additional_forward_args, args
):
    assert type(inputs) == torch.Tensor, \
        f'Only input type tensor supported, found {type(inputs)} instead.'
    name = explainer.get_name()
    
    # these methods don't support having multiple outputs at the same time
    if name in ['Deep Lift', 'Lime', 'Integrated Gradients', 'Gradient Shap']:
        attr_list = []
        for target in range(args.pred_len):
            score = explainer.attribute(
                inputs=inputs, baselines=baselines, target=target,
                additional_forward_args=additional_forward_args
            )
            attr_list.append(score)
        
        attr = torch.stack(attr_list)
        # pred_len x batch x seq_len x features -> batch x pred_len x seq_len x features
        attr = attr.permute(1, 0, 2, 3)
        
    elif name == 'Feature Ablation':
        attr = explainer.attribute(
            inputs=inputs, baselines=baselines,
            additional_forward_args=additional_forward_args
        )
    elif name == 'Occlusion':
        attr = explainer.attribute(
            inputs=inputs,
            baselines=baselines,
            sliding_window_shapes = (1,1),
            additional_forward_args=additional_forward_args
        )
    elif name == 'Augmented Occlusion':
        attr = explainer.attribute(
            inputs=inputs,
            sliding_window_shapes = (1,1),
            additional_forward_args=additional_forward_args
        )
    elif name == 'Morris Sensitivity':
        attr = explainer.attribute(
            inputs=inputs,
            additional_forward_args=additional_forward_args
        )
    else:
        print(f'{name} not supported.')
        raise NotImplementedError
    
    attr = attr.reshape(
        # batch x pred_len x seq_len x features
        (inputs.shape[0], args.pred_len, args.seq_len, attr.shape[-1])
    )
    
    return attr

def get_total_data(dataloader, device, add_x_mark=False):        
    if add_x_mark:
        return (
            torch.vstack([item[0] for item in dataloader]).float().to(device), 
            torch.vstack([item[2] for item in dataloader]).float().to(device)
        )
    else:
        return torch.vstack([item[0] for item in dataloader]).float().to(device)

def get_baseline(inputs, mode='random'):
    if type(inputs) == tuple:
        return tuple([get_baseline(input, mode) for input in inputs])
    
    batch_size = inputs.shape[0]    
    device = inputs.device
    
    if mode =='zero': baselines = torch.zeros_like(inputs, device=device).float()
    elif mode == 'random': baselines = torch.randn_like(inputs, device=device).float()
    elif mode == 'aug':
        means = torch.mean(inputs, dim=(0, 1))
        std = torch.std(inputs, dim=(0, 1))
        baselines = torch.normal(means, std).repeat(
            batch_size, inputs.shape[1], 1
        ).float()
    elif mode == 'mean': 
        baselines = torch.mean(
                inputs, axis=0
        ).repeat(batch_size, 1, 1).float()
    else:
        print(f'baseline mode options: [zero, random, aug, mean]')
        raise NotImplementedError
    
    return baselines

def normalize_feature_groups(df, features):
    summed = df[features].sum().T.reset_index()
    summed.columns = ['Feature', 'Score']
    summed['Score'] = summed['Score'] * 100 / summed['Score'].sum()
    print(summed)

def evaluate_interpretation(
    ground_truth:DataFrame, relevance_score:DataFrame, 
    features:List[Union[str , int]]
):
    merged = ground_truth.merge(
        relevance_score[['end_of_week'] + features], 
        on='end_of_week', how='inner'
    )
    
    true_features = [feature +'_x' for feature in features]
    pred_features = [feature +'_y' for feature in features]
    
    true_ranks = merged[true_features].rank(axis=1, ascending=False).reset_index(drop=True)
    predicted_ranks = merged[pred_features].rank(axis=1, ascending=False).reset_index(drop=True)
    # normalize the ranks
    y_true, y_pred = true_ranks/len(features), predicted_ranks/len(features)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html
    # Normalized Discounted Cumulative Gain.
    # This ranking metric returns a high value if true labels are ranked high by y_score.
    ndcg = ndcg_score(y_true, y_pred)
    
    true_scores = merged[true_features].div(
        merged[true_features].sum(axis=1), axis=0
    ).fillna(0) # when all are zero
    pred_scores = merged[pred_features].div(
        merged[pred_features].sum(axis=1), axis=0
    ).fillna(0)  # when all are zero
    
    normalized_mae = mean_absolute_error(true_scores, pred_scores)
    normalized_rmse = np.sqrt(mean_squared_error(true_scores, pred_scores))
    normalized_ndcg = ndcg_score(true_scores, pred_scores)

    print(f'Rank mae: {mae:0.5g}, rmse: {rmse:0.5g}, ndcg: {ndcg:0.5g}')
    print(f'Normalized mae: {normalized_mae:0.5g}, rmse: {normalized_rmse:0.5g}, ndcg: {normalized_ndcg:0.5g}')
    result_df = pd.DataFrame({
        'metrics':['mae', 'rmse', 'ndcg', 'normalized_mae', 'normalized_rmse', 'normalized_ndcg'],
        'values':[mae, rmse, ndcg, normalized_mae, normalized_rmse, normalized_ndcg]
    })
    return result_df

def find_first_common_date(group_cases, dates):
    six_days = pd.to_timedelta(6, unit='D')

    # dates needs to have at least 7 days of data includign the end_of_week
    for end_of_week in group_cases['end_of_week'].values:
        if end_of_week in dates and (end_of_week - six_days) in dates:
            print(f'Found first common date {end_of_week}.')
            return end_of_week
            
    print('Error. No overlapping dates found.')
    return None

def find_last_common_date(group_cases, dates):
    for date in group_cases['end_of_week'].values[::-1]:
        if date in dates:
            print(f'Found last common date {date}.')
            return date
    
    print('Error. No overlapping dates found.')
    return None

def aggregate_importance_by_window(
    df, features, end_of_week,
    window_size=7
):
    index = df[df['Date'] == end_of_week].first_valid_index()
    group_agg_scores = df[features].values
    weekly_sum = np.zeros(
        (group_agg_scores.shape[0] //window_size,  len(features)),
        dtype=np.float32
    )

    i = 0
    while index < group_agg_scores.shape[0]:
        weekly_sum[i] = np.sum(
            group_agg_scores[index - window_size + 1: index + 1], 
            axis=0
        )
        i += 1
        index += window_size
        
    weekly_agg_scores_df = pd.DataFrame()
    weekly_agg_scores_df['end_of_week'] = [
        end_of_week + pd.to_timedelta(i*window_size, unit='D') \
            for i in range(weekly_sum.shape[0])
    ]
    weekly_agg_scores_df[features] = weekly_sum
    
    return weekly_agg_scores_df

def calculate_global_rank(all_scores, features):
    data = pd.DataFrame(np.sum(all_scores, axis=-1), columns=features)
    summed = data[features].mean().T.reset_index()
    summed.columns = ['Feature', 'Score']
    summed['Score'] = summed['Score'] * 100 / summed['Score'].sum()
    
    print(summed)
    return summed

def align_interpretation(
    ranges:List, attr:np.ndarray, 
    features:List[Union[str, int]], min_date,  
    seq_len=14, pred_len=14
):
    pred_df = pd.DataFrame(ranges, columns=['FIPS', 'index'])
    # n_examples x features x pred_len -> n_examples x features
    horizons = list(range(pred_len))
    time_index_max = pred_df['index'].max()

    all_outputs = None
    for feature_index, feature in enumerate(features):
        pred_df[horizons] = attr[:, feature_index]
        groups = []
        
        for FIPS, group_df in pred_df.groupby('FIPS'):
            group_df.sort_values(by='index', inplace=True)
            new_df = pd.DataFrame({
                'index':[t +time_index_max for t in range(1, pred_len)],
                'FIPS':FIPS
            })
            new_df[horizons] = np.nan
            new_df = new_df[group_df.columns]
            group_df = pd.concat([group_df, new_df], axis=0).reset_index(drop=True)
            
            for horizon in horizons:
                    group_df[horizon] = group_df[horizon].shift(periods=horizon, axis=0)
                    
            group_df[feature] = group_df[horizons].mean(axis=1, skipna=True)
            groups.append(group_df.drop(columns=horizons))
        
        groups = pd.concat(groups, axis=0)
        
        if all_outputs is None: all_outputs = groups
        else:
            all_outputs = all_outputs.merge(
                groups, how='inner', on=['FIPS', 'index']
            )

    all_outputs[features] = all_outputs[features].div(all_outputs[features].sum(axis=1), axis=0)

    all_outputs['Date'] = min_date + pd.to_timedelta(
        seq_len + all_outputs['index'], unit='D')
    all_outputs.drop(columns='index', inplace=True)

    print(all_outputs['Date'].min(), all_outputs['Date'].max())
    return all_outputs