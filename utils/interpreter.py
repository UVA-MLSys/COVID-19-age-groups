import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import ndcg_score
from pandas import DataFrame

def align_feature_scores(
        data, time_index, last_time_step, 
        time_range, scores, features
    ):
    df = data[data[time_index]<=last_time_step][['Date', 'FIPS']]
    
    # n_samples x features x pred_len
    if len(scores.shape) == 3:
        df[features] = np.sum(scores, axis=-1)
    
    # n_samples x features
    else:
        df[features] = scores
    
    df.dropna(inplace=True)
    print(df[features].sum().T)
    return df

def evaluate_interpretation(
    ground_truth:DataFrame, relevance_score:DataFrame, features:List[str | int]
):
    merged = ground_truth.merge(
        relevance_score[['end_of_week'] + features], 
        on='end_of_week', how='inner'
    )
    
    y_true = merged[[feature +'_x' for feature in features]]
    y_score = merged[[feature+'_y' for feature in features]]
    
    ndcg = ndcg_score(y_true, y_score)
    print(f'ndcg score {ndcg:0.5g}')
    return ndcg

def find_first_common_date(group_cases, dates):
    one_week = pd.to_timedelta(7, unit='D')

    for end_of_week in group_cases['end_of_week'].values:
        if end_of_week in dates and (end_of_week - one_week) in dates:
            print(f'Found first common date {end_of_week}.')
            return end_of_week
            
    print('Error. No overlapping dates found.')
    return None

def aggregate_importance_by_window(
    df, features, end_of_week,
    window_size=7
):
    index = df[df['Date'] == end_of_week].first_valid_index()
    group_agg_scores = df[features].values
    weekly_sum = np.zeros((group_agg_scores.shape[0] //window_size,  len(features)))

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

def calculate_global_rank(df, all_scores, features):
    data = df.copy()
    
    data[features] = np.sum(all_scores, axis=-1) 
    summed = data[features].mean().T.reset_index()
    summed.columns = ['Feature', 'Score']
    summed['Score'] = summed['Score'] * 100 / summed['Score'].sum()
    
    print(summed)
    return summed

def align_interpretation(
        df:pd.DataFrame, all_scores:np.ndarray, features:List[str|int]
    ):
    num_dates = df['Date'].nunique()
    num_group_ids = df['FIPS'].nunique()
    pred_len = all_scores.shape[-1]

    group_agg_scores = np.full(
        (num_dates + pred_len, len(features), pred_len), 
        fill_value=np.nan, dtype=np.float16
    )

    for feature_index in range(len(features)):
        time_delta = 0
        index = 0
        while index < df.shape[0]:
            group_agg_scores[time_delta, feature_index, :] = np.sum(
                all_scores[
                    index:(index + num_group_ids), feature_index, :
                ], axis=0
            )
            
            index += num_group_ids 
            time_delta += 1
            
    for horizon in range(pred_len):
        group_agg_scores[:, :, horizon] = np.roll(
            group_agg_scores[:, :, horizon], 
            shift=horizon+1, axis=0
        )
    group_agg_scores = group_agg_scores[1:]
    
    group_agg_scores_df = pd.DataFrame()
    group_agg_scores_df['Date'] = [
        df['Date'].min() + pd.to_timedelta(i, unit='D') \
        for i in range(1, group_agg_scores.shape[0] + 1)
    ]
    group_agg_scores_df[features] = np.nanmean(
        group_agg_scores, axis=-1
    )
    
    return group_agg_scores_df