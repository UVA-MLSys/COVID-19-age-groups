import pandas as pd
from pandas import DataFrame, to_datetime, to_timedelta
import numpy as np
from typing import List
import os, gc, random, shutil
from torch import Tensor

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, explained_variance_score
from pytorch_lightning import seed_everything

from data.data_factory import AgeData

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