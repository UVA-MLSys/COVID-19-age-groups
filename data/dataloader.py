from typing import Any, Callable, Dict, List, Tuple, Union, Optional
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer

from experiment.config import Split

class AgeDataLoader:
    def __init__(
        self, data_path:str, date_index:str, seq_len:int, 
        pred_len:int, group_ids:List[str] = [],
        static_reals:List[str] = [], static_categoricals:List[str] = [],
        observed_reals:List[str] = [], 
        observed_categoricals:List[str] = [],
        known_reals:List[str] = [], known_categoricals:List[str] = [],
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
            static_categoricals (List[str]): list of categorical variables that do not change over time,
                entries can be also lists which are then encoded together.
            observed_reals (List[str]): list of continuous variables that change over
                time and are not known in the future. You might want to include your target here.
            observed_categoricals (List[str]): list of categorical variables that change over
                time and are not known in the future, entries can be also lists which are then encoded together
                (e.g. useful for weather categories). You might want to include your target here.
            known_reals (List[str]): list of continuous variables that change over
                time and are known in the future (e.g. price of a product, but not demand of a product).
            known_categoricals (List[str]): list of categorical variables that change over
                time and are known in the future, entries can be also lists which are then encoded together
                (e.g. useful for special days or promotion categories).
            targets (List[str]): column denoting the target or list of columns denoting the target -
                categorical or continous.
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
        self.static_categoricals = static_categoricals

        self.observed_reals = observed_reals
        self.observed_categoricals = observed_categoricals
        self.known_reals = known_reals
        self.known_categoricals = known_categoricals
        
        self.targets = targets
        
        selected_columns = [date_index] + group_ids + static_reals + \
            static_categoricals + observed_reals + observed_categoricals + \
            known_reals + known_categoricals + targets
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
        self.categorical_feature_scaler:Optional[LabelEncoder] = None
    
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

        print(f'Train samples {train_data.shape[0]}, validation samples {val_data.shape[0]}, test samples {test_data.shape[0]}')

        train_days = (split.val_start - split.train_start).days
        validation_days = (split.test_start - split.val_start).days
        test_days = (split.test_end - split.test_start).days + 1

        print(f'{train_days} days of training, {validation_days} days of validation data, {test_days} days of test data.')
        
        if self.scale:
            self.fit_scalers(train_data)

        return train_data, val_data, test_data
    
    def fit_scalers(self, train_data:DataFrame):
        print('Fitting scalers on train data')
        # fit scalers
        real_features = self.static_reals + self.observed_reals + self.known_reals
        # targets are separately scaled. targets can be observed features
        real_features =  [feature for feature in real_features if feature not in self.targets]
        
        categorical_features = self.static_categoricals + self.observed_categoricals + self.known_categoricals
        if len(real_features) > 0:
            self.real_feature_scaler = StandardScaler().fit(train_data[real_features])
        if len(categorical_features):
            self.categorical_feature_scaler = LabelEncoder().fit(train_data[categorical_features])
        
        # this work has only real values as target
        # TODO: make more generalized by supporing both categorical and real scaling here
        self.target_scaler = StandardScaler().fit(train_data[self.targets])
        
    def _scale_data(self, df:DataFrame) -> DataFrame:        
        data = df.copy()
        # fit scalers
        real_features = self.static_reals + self.observed_reals + self.known_reals
        # targets are separately scaled. targets can be observed features
        real_features =  [feature for feature in real_features if feature not in self.targets]
        
        categorical_features = self.static_categoricals + self.observed_categoricals + self.known_categoricals
        
        if self.real_feature_scaler:
            data[real_features] = self.real_feature_scaler.transform(
                data[real_features]
            )
        if self.categorical_feature_scaler:
            data[categorical_features] = self.categorical_feature_scaler.transform(
                data[categorical_features]
            )
        if self.target_scaler:
            data[self.targets] = self.target_scaler.transform(data[self.targets])
            
        return data
        
    def upscale_target(self, data:DataFrame):
        if self.target_scaler is None:
            return data
        
        data[self.targets] = self.target_scaler.inverse_transform(
            data[self.targets]
        )
        return data
    
    def create_timeseries(
        self, data:DataFrame, train:bool=False
    ) -> Tuple[TimeSeriesDataSet, DataLoader]:
        
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
            static_categoricals = self.static_categoricals,
            time_varying_unknown_reals=self.observed_reals,
            time_varying_unknown_categoricals=self.observed_categoricals,
            time_varying_known_reals = self.known_reals,
            time_varying_known_categoricals = self.known_categoricals,
            # add_target_scales=True,
            target_normalizer = MultiNormalizer(
                [GroupNormalizer(groups=self.group_ids) for _ in self.targets]
            )
        )

        batch_size = self.batch_size
        if type(batch_size)==list:
            assert len(batch_size) >= 2, ValueError("Batch list must be of size 2. [train batch, val/test batch]")
            batch_size = batch_size[0] if train else batch_size[1]
        
        dataloader = data_timeseries.to_dataloader(train=train, batch_size=batch_size)
        
        return data_timeseries, dataloader 