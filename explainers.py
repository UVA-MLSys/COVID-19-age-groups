import abc, gc
import numpy as np
from tqdm import tqdm
from data.dataloader import AgeDataLoader
import pandas as pd
from typing import List

class BaseExplainer(abc.ABC):
    """
    A base class for explainer.
    """
    
    def __init__(self, model, dataloader:AgeDataLoader, features:List[str]) -> None:
        self.model = model
        self.loader = dataloader
        self.features =  features
        
        self.time_index = dataloader.time_index
        self.pred_len = dataloader.pred_len
        self.seq_len = dataloader.seq_len
        self.targets = dataloader.targets
        
        assert len(self.targets) == 1, \
            ValueError("Explainers aren't yet implemented for multiple targets")
        
    def train_generators(self, data):
        """
        Train and save any values necessary for later usage. 
        This method is called before starting any attribute calculation.
        """
    
    @abc.abstractmethod
    def _local_attribute(self, data, x, t, y_t, all_scores):
        """
        Generate local interpretation
        """
        
    def time_range(self, df):
        return range(
            df[self.time_index].min() + self.seq_len - 1, 
            df[self.time_index].max() - self.pred_len + 1
        )
    
    def attribute(self, df:pd.DataFrame, disable_progress=False):
        """
        Calculate the feature attributes
        
        Args:
            data : input dataframe
        """
        data = df.copy()
        time_index = self.time_index
        features = self.features
        time_range = self.time_range(data)
        
        # this is necessary for later score alignment
        data = data.sort_values(by=time_index).reset_index(drop=True)
        self.train_generators(data)
        
        all_scores = np.full(
            (
                data[data[time_index]<=time_range[-1]].shape[0], 
                len(features), self.pred_len
            ), fill_value=np.nan, dtype=np.float16
        )

        for t in tqdm(time_range, disable=disable_progress):
            # temporal slice [t- seq_len + 1:t + pred_len]
            x = data[
                (data[time_index]  > t - self.seq_len) & 
                (data[time_index] <= (t + self.pred_len))
            ]
            
            _, dataloader = self.loader.create_timeseries(x)
            y_t = self.model.predict(dataloader) 
            
            self._local_attribute(x, t, y_t, all_scores)
        
        # since we are trying to calculate the importance in future 
        # feature importance are calculated from minimum time seq_len-1
        # this drops those early nan values
        num_group_ids = data[self.loader.group_ids].nunique().max()
        all_scores = all_scores[num_group_ids*(self.seq_len-1):]
        
        gc.collect()
        return all_scores
        
    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the explainer.
        """
        return "Base"
        
class FeatureOcclusion(BaseExplainer):
    """
    Feature Occlusion (FO) [1] computes attribution as the difference
    in output after replacing each contiguous region with a given baseline. For time series we
    considered continuous regions as features with in same time step or multiple time steps
    grouped together.
    
    [1] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In
    European conference on computer vision, 2014.
    """
    def __init__(
        self, model, dataloader:AgeDataLoader, 
        features:List[str], n_samples:int=2
    ) -> None:
        super().__init__(model, dataloader, features)
        self.n_samples = n_samples
        
    def train_generators(self, data):
        """
        Train and save any values necessary for later usage. 
        This method is called before starting any attribute calculation.
        """
        self.minimums = data[self.features].min().values
        self.maximums = data[self.features].max().values
        
    def _local_attribute(self, x, t, y_t, all_scores):
        for feature_index, feature in enumerate(self.features):
            selected_index = x[x[self.time_index] <= t].index
            
            x_hat = x.copy()
            x_hat.loc[selected_index, feature] = np.random.uniform(
                self.minimums[feature_index], self.maximums[feature_index], 
                size = selected_index.shape[0]
            )
            
            _, dataloader = self.loader.create_timeseries(x_hat)
            y_hat_t = self.model.predict(dataloader)
            
            diff = abs(y_t[0] - y_hat_t[0]).detach().cpu().numpy()
            
            # no need to return as arrays are passed by reference
            all_scores[x[x[self.time_index] == t].index, feature_index, :] = diff
    
    def get_name(self) -> str:
        return "Feature Occlusion"
    
class AugmentedFeatureOcclusion(BaseExplainer):
    """
    Augmented Feature Occlusion (AFO) [1], where we replace observations
    with samples from the bootstrapped distribution p(xi) for each feature i. This is to avoid generating
    out-of-distribution noise samples.
    
    [1] Sana Tonekaboni, Shalmali Joshi, Kieran Campbell, David K Duvenaud, and Anna Goldenberg.
    What went wrong and when? Instance-wise feature importance for time-series black-box models.
    In Neural Information Processing Systems, 2020.
    """
    
    def __init__(
        self, model, dataloader:AgeDataLoader, 
        features:List[str], n_samples:int=2
    ) -> None:
        super().__init__(model, dataloader, features)
        self.n_samples = n_samples
        
    def train_generators(self, data):
        self.dist = data[self.features].values
        
    def _local_attribute(self, x, t, y_t, all_scores):
        for feature_index, feature in enumerate(self.features):
            selected_index = x[x[self.time_index] <= t].index
            
            x_hat = x.copy()
            x_hat.loc[selected_index, feature] = np.random.choice(
                self.dist[feature_index],
                size = selected_index.shape[0]
            )
            
            _, dataloader = self.loader.create_timeseries(x_hat)
            y_hat_t = self.model.predict(dataloader)
            
            diff = abs(y_t[0] - y_hat_t[0]).detach().cpu().numpy()
            
            # no need to return as arrays are passed by reference
            all_scores[x[self.time_index] == t, feature_index, :] = diff
    
    def get_name(self) -> str:
        return "Feature Occlusion"
    
class FeatureAblation(BaseExplainer):
    """
        Feature Ablation (FA) [1] [2] computes attribution as the difference in
        output after replacing each feature with a baseline. Input features can also be grouped
        and ablated together rather than individually.
        
        [1] Harini Suresh, Nathan Hunt, Alistair Johnson, Leo Anthony Celi, Peter Szolovits, and Marzyeh
        Ghassemi. Clinical intervention prediction and understanding using deep networks. arXiv
        preprint arXiv:1705.08498, 2017.
        
        [2] Ozan Ozyegen, Igor Ilic, and Mucahit Cevik. Evaluation of interpretability methods for multivariate
        time series forecasting. Applied Intelligence, pages 1–17, 2022.
    """
    def __init__(
        self, model, dataloader:AgeDataLoader, 
        features:List[str], method:str="global"
    ) -> None:
        """
        Args:
            method (str): Ablation method. Options: global, local. Defaults to "global".
        """        """"""
        super().__init__(model, dataloader, features)
        
        assert method in ['global', 'local'], "Unknown method type"
        self.method = method
        
    def train_generators(self, data):
        if self.method == "global":
            self.baselines = data[self.features].mean().values
        
    def _local_attribute(self, x, t, y_t, all_scores):
        for feature_index, feature in enumerate(self.features):
            selected_index = x[x[self.time_index] <= t].index
            
            x_hat = x.copy()
            
            if self.method == "global":
                x_hat.loc[selected_index, feature] = self.baselines[feature_index]
            else:
                x_hat.loc[selected_index, feature] = x_hat[selected_index][feature].mean()
            
            _, dataloader = self.loader.create_timeseries(x_hat)
            y_hat_t = self.model.predict(dataloader)
            
            diff = abs(y_t[0] - y_hat_t[0]).detach().cpu().numpy()
            
            # no need to return as arrays are passed by reference
            all_scores[x[self.time_index] == t, feature_index, :] = diff
    
    def get_name(self) -> str:
        return "Feature Ablation"