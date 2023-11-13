import pandas as pd
from pandas import DataFrame, to_datetime, to_timedelta
import numpy as np
from typing import List
import os, gc, random, shutil
from torch import Tensor

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, explained_variance_score
from pytorch_lightning import seed_everything

def align_predictions(
    ground_truth:DataFrame, predictions_index:DataFrame, 
    predictions:List, dataloader,
    remove_negative:bool=True, upscale:bool=True
):
    horizons = range(dataloader.pred_len)
    time_index_max = predictions_index[dataloader.time_index].max()

    targets, time_index, group_ids = dataloader.targets, dataloader.time_index, dataloader.group_ids
    # a groupby with a groupength 1 throws warning later
    if len(group_ids) == 1: group_ids = group_ids[0]
    
    all_outputs = None 
    for target_index, target in enumerate(targets):
        if type(predictions[target_index]) == Tensor:
            predictions[target_index] = predictions[target_index].numpy()

        pred_df = DataFrame(
            predictions[target_index], columns=horizons
        )
        pred_df = pd.concat([predictions_index, pred_df], axis=1)
        outputs = []

        for group_id, group_df in pred_df.groupby(group_ids):
            group_df = group_df.sort_values(
                by=time_index
            ).reset_index(drop=True)

            new_df = DataFrame({
                time_index : [t + time_index_max for t in range (1, dataloader.pred_len)]
            })
            new_df[group_ids] = group_id
            new_df.loc[:, horizons] = None
            new_df = new_df[group_df.columns]
            group_df = pd.concat([group_df, new_df], axis=0).reset_index(drop=True)

            for horizon in horizons:
                group_df[horizon] = group_df[horizon].shift(periods=horizon, axis=0)
                
            group_df[target] = group_df[horizons].mean(axis=1, skipna=True)
            outputs.append(group_df.drop(columns=horizons))

        outputs = pd.concat(outputs, axis=0)
        
        if all_outputs is None: all_outputs = outputs
        else: 
            all_outputs = all_outputs.merge(
                outputs, how='inner', on=list(predictions_index.columns)
            )
            
    gc.collect()
      
    # upscale the target values if needed
    if upscale:
        all_outputs = dataloader.upscale_target(all_outputs)
        
    # must appear after upscaling
    if remove_negative:
        # remove negative values, since infection count can't be negative
        for target in targets:
            all_outputs.loc[all_outputs[target]<0, target] = 0
            
    # add `Predicted` prefix to the predictions
    all_outputs.rename(
        {target:'Predicted_'+target for target in targets}, 
        axis=1, inplace=True
    )
    
    # only keep the directly relevant columns
    ground_truth = ground_truth[[dataloader.date_index] + list(predictions_index.columns) + targets]
    # merge with grounth truth for evaluation
    all_outputs = ground_truth.merge(
        all_outputs, how='inner', on=list(predictions_index.columns)
    )
    
    return all_outputs

def add_day_time_features(dates):
    df_stamp = pd.DataFrame({'date': dates})

    # Time series library day encoding takes 3 values
    # check layers.Embed.TimeFeatureEmbedding class
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    
    return df_stamp.drop(columns=['date'])

def clear_directory(dir_path):
    if os.path.exists(dir_path):
        assert os.path.isdir(dir_path), ValueError(f"The provided path {dir_path} isn't a directory.")
        
        print(f'Directory {dir_path} already exists ! Removing ...')
        shutil.rmtree(dir_path)
    else:
        print(f"Directory {dir_path} doesn't exists ! Creating ...")
    os.makedirs(dir_path, exist_ok=True)

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    seed_everything(seed, workers=True)

def get_best_model_path(checkpoint_folder, prefix='best-epoch='):
    for item in os.listdir(checkpoint_folder):
        if item.startswith(prefix):
            print(f'\nFound best checkpoint model {item}.')
            return os.path.join(checkpoint_folder, item)

    raise FileNotFoundError(f"Couldn't find the best model in {checkpoint_folder}")

def calculate_result(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    # smape = symmetric_mean_absolute_percentage(y_true, y_pred)
    # nnse = normalized_nash_sutcliffe_efficiency(y_true, y_pred)
    r2 = explained_variance_score(y_true, y_pred)

    return mae, rmse, rmsle, r2

def remove_outliers(
    original_df, multiplier:int=3, 
    verbose:bool=False, window:int=7
):
    df = original_df.copy()

    date_columns = sorted([col for col in df.columns if valid_date(col)])
    total_outliers = 0
    fips = df['FIPS'].values

    for i in range(df.shape[0]):
        county_data = df.loc[i, date_columns]
        
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
        window = county_data.rolling(window=7)
        
        median = window.quantile(0.5)
        q1 = window.quantile(0.25) 
        q3 = window.quantile(0.75)

        iqr = q3-q1
        upper_limit = q3 + multiplier*iqr
        lower_limit = q1 - multiplier*iqr

        # Alternatives to consider, median > 0 or no median condition at all
        higher_outliers = (county_data > upper_limit) & (median >= 0)
        lower_outliers = (county_data < lower_limit) & (median >= 0)

        number_of_outliers = sum(higher_outliers) + sum(lower_outliers)
        total_outliers += number_of_outliers

        # when no outliers found
        if number_of_outliers == 0: continue
        if verbose:
            print(f'FIPS {fips[i]}, outliers found, higher: {county_data[higher_outliers].shape[0]}, lower: {county_data[lower_outliers].shape[0]}.')

        county_data[higher_outliers] = upper_limit
        county_data[lower_outliers] = lower_limit
        
        df.loc[i, date_columns] = county_data
    
    print(f'Outliers found {total_outliers}, percent {total_outliers*100/(df.shape[0]*len(date_columns)):.3f}')
    return df

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

# https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
# NSE is equivalent to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score
def normalized_nash_sutcliffe_efficiency(y_true, y_pred):
    # numerator = sum (np.square(y_true - y_pred) )
    # denominator = sum(np.square(y_true - np.mean(y_true)))
    # NSE = 1 - numerator / denominator
    NSE = explained_variance_score(y_true, y_pred)
    return 1 / ( 2 - NSE)

# https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.metrics.point.SMAPE.html?highlight=smape
def symmetric_mean_absolute_percentage(y_true, y_pred):
    value = 2*abs(y_true - y_pred) / abs(y_true) + abs(y_pred)
    
    # for cases when both ground truth and predicted value are zero
    value = np.where(np.isnan(value), 0, value)
    
    return np.mean(value)

def show_result(df: DataFrame, targets:List[str]):    
    for target in targets:
        predicted_column = f'Predicted_{target}'
        y_true, y_pred = df[target].values, df[predicted_column].values

        mae, rmse, rmsle, r2 = calculate_result(y_true, y_pred)
        print(f'Target {target}: MAE {mae:.5g}, RMSE {rmse:.5g}, RMSLE {rmsle:0.5g}, R2 {r2:0.5g}.')
    print()

def read_feature_file(dataPath, file_name):
    df = pd.read_csv(os.path.join(dataPath, file_name))
    # drop empty column names in the feature file
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def valid_date(date):
    try:
        to_datetime(date)
    except:
        return False
    return True

def convert_cumulative_to_daily(original_df):
    df = original_df.copy()

    date_columns = [col for col in df.columns if valid_date(col)]
    df_advanced = df[date_columns].shift(periods=1, axis=1, fill_value=0)
    df[date_columns] -= df_advanced[date_columns]
    return df

def missing_percentage(df):
    return df.isnull().mean().round(4).mul(100).sort_values(ascending=False)