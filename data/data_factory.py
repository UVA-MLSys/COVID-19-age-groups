from typing import Any, Callable, Dict, List, Tuple, Union, Optional
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
import pandas as pd, numpy as np
from torch.utils.data import DataLoader, Dataset
from pytorch_forecasting.data import TimeSeriesDataSet
import os
from data.dataloader import MultiTimeSeries
from exp.config import Split, DataConfig

class AgeData:
    @staticmethod
    def build(args):
        return AgeData(
            data_path=os.path.join(args.root_path, args.data_path),
            date_index=DataConfig.date_index, 
            seq_len=args.seq_len, pred_len=args.pred_len,
            group_ids=DataConfig.group_ids, 
            static_reals=DataConfig.static_reals,
            observed_reals=DataConfig.observed_reals,
            known_reals=DataConfig.known_reals,
            targets=DataConfig.targets,
            scale=not args.no_scale
        )
    
    def __init__(
        self, data_path:str, date_index:str, seq_len:int, 
        pred_len:int, group_ids:List[str] = [],
        static_reals:List[str] = [],
        observed_reals:List[str] = [], 
        known_reals:List[str] = [],
        targets:List[str] = [], scale:bool=True,
        batch_size:Union[int, List[int]] = [64, 256]
    ):
        """Configuration class for TimeSeriesDataset at 
        https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html# 

        Args:
            data_path (str): path of the input file
            date_index (str): date column denoting the time. This column 
                is used to calculate the time index and determine the sequence of samples.
            seq_len (int): input window length.
            pred_len (int): output prediction length.
            group_ids (List[str]): list of column names identifying a time series. This means 
                that the ``group_ids`` identify a sample together with the ``time_idx``. 
                If you have only one timeseries, set this to the name of column that is constant.
            static_reals (List[str]): list of continuous variables that do not change over time.
            observed_reals (List[str]): list of continuous variables that change over
                time and are not known in the future. You might want to include your target here.
            known_reals (List[str]): list of continuous variables that change over
                time and are known in the future (e.g. price of a product, but not demand of a product).
            targets (List[str]): column denoting the target or list of columns denoting the continous target.
            scale (bool): whether to scale the input and target features. Default True.
            batch_size (Union[int, List[int]]): batch size of the dataset. If a list is provided, the size at the 
                first index is used in the train data and the second index is used for test/validation data.
        """
        self.data_path = data_path

        # input and output features
        self.group_ids = group_ids
        self.time_index = 'TimeFromStart'
        self.date_index = date_index
        self.static_reals = static_reals

        self.observed_reals = observed_reals
        self.known_reals = known_reals
        
        self.real_features = static_reals + observed_reals + known_reals
        
        self.targets = targets
        
        selected_columns = [date_index] + group_ids + static_reals + \
            observed_reals + known_reals + targets
        # remove any duplicates
        self.selected_columns = []
        for column in selected_columns:
            if column not in self.selected_columns:
                self.selected_columns.append(column)

        # sizes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        
        # scalers
        self.scale = scale
        self.target_scaler:Optional[StandardScaler] = None
        self.real_feature_scaler:Optional[StandardScaler] = None
    
    def read_df(self):
        df = pd.read_csv(self.data_path)[self.selected_columns]
        df[self.date_index] = pd.to_datetime(df[self.date_index])

        return df
        
    def split_data(self, df:DataFrame, split:Split):
        dates = df[self.date_index]
        df[self.time_index] = (dates - split.train_start).dt.days
        train_data = df[(dates >= split.train_start) & (dates < split.val_start)]
        
        # at least input_sequence_length prior days data is needed to start prediction
        # this ensures prediction starts from date validation_start. 
        val_start = split.val_start - pd.to_timedelta(self.seq_len, unit='day') 
        val_data = df[(dates >= val_start) & (dates < split.test_start)]

        test_start = split.test_start - pd.to_timedelta(self.seq_len, unit='day')
        test_data = df[(dates >= test_start) & (dates <= split.test_end)]

        print(f'\nTrain samples {train_data.shape[0]}, validation samples {val_data.shape[0]}, test samples {test_data.shape[0]}')

        train_days = (split.val_start - split.train_start).days
        validation_days = (split.test_start - split.val_start).days
        test_days = (split.test_end - split.test_start).days + 1

        print(f'{train_days} days of training, {validation_days} days of validation data, {test_days} days of test data.\n')
        
        if self.scale:
            self.fit_scalers(train_data)

        return train_data, val_data, test_data
    
    def fit_scalers(self, train_data:DataFrame):
        print('Fitting scalers on train data')
        # targets are separately scaled. targets can be observed features
        real_input_features =  [
            feature for feature in self.real_features \
                if feature not in self.targets
        ]
        
        if len(real_input_features) > 0:
            self.real_feature_scaler = StandardScaler().fit(train_data[real_input_features])
        
        # this work has only real values as target
        # TODO: make more generalized by supporing both categorical and real scaling here
        self.target_scaler = StandardScaler().fit(train_data[self.targets])
        
    def _scale_data(self, df:DataFrame) -> DataFrame:        
        data = df.copy()
        # targets are separately scaled. targets can be observed features
        real_input_features =  [
            feature for feature in self.real_features \
                if feature not in self.targets
        ]
        
        if self.real_feature_scaler:
            data[real_input_features] = self.real_feature_scaler.transform(
                data[real_input_features]
            )

        if self.target_scaler:
            data[self.targets] = self.target_scaler.transform(data[self.targets])
            
        return data
        
    def upscale_target(self, data:Union[DataFrame, np.ndarray]):
        if self.target_scaler is None: return data
        
        if type(data) == DataFrame:
            data[self.targets] = self.target_scaler.inverse_transform(
                data[self.targets]
            )
        elif len(data.shape) == 2: 
            # n_examples x n_targets
            data = self.target_scaler.inverse_transform(data)
        elif len(data.shape) == 3 and data.shape[-1] == len(self.targets): 
            # n_examples x pred_len x n_targets
            original_shape = data.shape
            data = data.reshape((-1, len(self.targets)))
            data = self.target_scaler.inverse_transform(data)
            data = data.reshape(original_shape)
            
        return data
    
    def create_timeseries(
        self, data:DataFrame, train:bool=False
    ) -> Tuple[Dataset, DataLoader]:
        
        if self.scale:
            data = self._scale_data(data)
        
        # Note that TimeSeriesDataSet by default uses StandardScaler on the real values
        data_timeseries = TimeSeriesDataSet(
            data,
            time_idx = self.time_index,
            target = self.targets,
            group_ids = self.group_ids, 
            max_encoder_length = self.seq_len,
            max_prediction_length = self.pred_len,
            static_reals = self.static_reals,
            time_varying_unknown_reals=self.observed_reals,
            time_varying_known_reals = self.known_reals
        )

        batch_size = self.batch_size
        if type(batch_size)==list:
            assert len(batch_size) >= 2, ValueError("Batch list must be of size 2. [train batch, val/test batch]")
            batch_size = batch_size[0] if train else batch_size[1]
        
        dataloader = data_timeseries.to_dataloader(train=train, batch_size=batch_size)
        
        return data_timeseries, dataloader 
    
    def create_tslib_timeseries(
        self, data:DataFrame, train:bool=False, ts_dataset=None
    ) -> Tuple[MultiTimeSeries, DataLoader]:
        if self.scale:
            data = self._scale_data(data)
            
        if ts_dataset is None:
            ts_dataset = MultiTimeSeries(
                data, self.seq_len, self.pred_len, 
                self.static_reals + self.observed_reals, 
                self.known_reals, self.targets
            )
        if type(self.batch_size) == list:
            assert len(self.batch_size) == 2, \
                'Batch size list can have two items at most. [train batch, val/test batch]'
            if train: batch_size = self.batch_size[0]
            else: batch_size = self.batch_size[1]
        else:
            batch_size = self.batch_size
        
        ts_dataloader = DataLoader(
            ts_dataset, batch_size, shuffle=train
        )
        return ts_dataset, ts_dataloader