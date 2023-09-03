import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import dcg_score, ndcg_score, mean_absolute_error, mean_squared_error
from pandas import DataFrame

def normalize_feature_groups(df, features):
    summed = df[features].sum().T.reset_index()
    summed.columns = ['Feature', 'Score']
    summed['Score'] = summed['Score'] * 100 / summed['Score'].sum()
    print(summed)

def evaluate_interpretation(
    ground_truth:DataFrame, relevance_score:DataFrame, features:List[str | int]
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
    
    print(f'Rank mae: {mae:0.5g}, rmse: {rmse:0.5g}, ndcg: {ndcg:0.5g}')
    
    true_scores = merged[true_features].div(merged[true_features].sum(axis=1), axis=0)
    pred_scores = merged[pred_features].div(merged[pred_features].sum(axis=1), axis=0)
    
    normalized_mae = mean_absolute_error(true_scores, pred_scores)
    normalized_rmse = np.sqrt(mean_squared_error(true_scores, pred_scores))
    normalized_ndcg = ndcg_score(true_scores, pred_scores)

    print(f'Normalized mae: {normalized_mae:0.5g}, rmse: {normalized_rmse:0.5g}, ndcg: {normalized_ndcg:0.5g}')
    return mae, rmse, ndcg, normalized_mae, normalized_rmse, normalized_ndcg

def find_first_common_date(group_cases, dates):
    six_days = pd.to_timedelta(6, unit='D')

    # dates needs to have at least 7 days of data includign the end_of_week
    for end_of_week in group_cases['end_of_week'].values:
        if end_of_week in dates and (end_of_week - six_days) in dates:
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
        fill_value=np.nan, dtype=np.float32
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