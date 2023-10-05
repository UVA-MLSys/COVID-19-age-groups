import numpy as np
from typing import List
import SALib, torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

class MorrisSensitivty:
    def __init__(self, model, inputs, pred_len) -> None:
        self.model = model
        self.pred_len = pred_len
        
        self.sp = self._build_problem_spec(inputs)
        
    def _build_problem_spec(self, inputs:torch.Tensor):
        bounds = []
        dists = []
        n_features = inputs.shape[-1] # batch x seq_len x features
        for i in range(n_features):
            bounds.append([
                torch.min(inputs[:, :, i]).item(), 
                torch.max(inputs[:, :, i]).item()
            ])
            # for now SALib throws error or returns np.inf for any other distribution
            # other supported distributions: norm, triang, lognorm
            # https://salib.readthedocs.io/en/latest/user_guide/advanced.html#generating-alternate-distributions
            dists.append('unif')
            
        sp = SALib.ProblemSpec({
            "num_vars": n_features,
            'bounds': bounds,
            'dists': dists,
            # 'sample_scaled': True
        })
        return sp
        
    def attribute(
        self, inputs:torch.Tensor, 
        additional_forward_args:TensorOrTupleOfTensorsGeneric
    ):
        (batch_size, seq_len, n_features) = inputs.shape
        samples = SALib.sample.morris.sample(self.sp, batch_size)
        samples_reshaped = samples.reshape((-1, batch_size, n_features))
        
        morris_iterations = samples_reshaped.shape[0]
        pred_len = self.pred_len
        device = inputs.device
    
        # batch x pred_len x seq_len x features
        attr = torch.zeros(size = (batch_size, pred_len, seq_len, n_features))
        y_hats = np.zeros(shape=(morris_iterations, batch_size, pred_len, 1))
        samples_reshaped = torch.tensor(samples_reshaped, device=device)

        for t in range(seq_len):
            x_hat = inputs.clone()
            
            for morris_itr in range(morris_iterations):
                x_hat[:, t] = samples_reshaped[morris_itr]
                y_hat = self.model(x_hat, *additional_forward_args)
                y_hats[morris_itr] = y_hat.detach().cpu().numpy()
                
            y_hats_reshaped = y_hats.reshape((-1, pred_len))
            for pred_index, Y in enumerate(y_hats_reshaped.T):
                morris_index = SALib.analyze.morris.analyze(
                    self.sp, samples, Y
                )['mu_star'].data
                attr[:, pred_index, t] = torch.tensor(morris_index, device=device)
        
        return attr
        
    def get_name(self):
        return 'Morris Sensitivity'

# import abc, gc

# from tqdm import tqdm
# from data.data_factory import AgeData
# import pandas as pd

# class BaseExplainer(abc.ABC):
#     """
#     A base class for explainer.
#     """
    
#     def __init__(self, model, dataloader:AgeData, features:List[str]) -> None:
#         self.model = model
#         self.dataloader = dataloader
#         self.features =  features
        
#         self.time_index = dataloader.time_index
#         self.pred_len = dataloader.pred_len
#         self.seq_len = dataloader.seq_len
#         self.targets = dataloader.targets
        
#         assert len(self.targets) == 1, \
#             ValueError("Explainers aren't yet implemented for multiple targets")
        
#     def train_generators(self, data):
#         """
#         Train and save any values necessary for later usage. 
#         This method is called before starting any attribute calculation.
#         """
    
#     @abc.abstractmethod
#     def _local_attribute(self, data, x, t, y_t, all_scores):
#         """
#         Generate local interpretation
#         """
        
#     def time_range(self, df):
#         return range(
#             df[self.time_index].min() + self.seq_len - 1, 
#             df[self.time_index].max() - self.pred_len + 1
#         )
    
#     def attribute(self, df:pd.DataFrame, disable_progress=False):
#         """
#         Calculate the feature attributes
        
#         Args:
#             data : input dataframe
#         """
#         data = df.copy()
#         time_index = self.time_index
#         features = self.features
#         time_range = self.time_range(data)
        
#         # this is necessary for later score alignment
#         data = data.sort_values(
#             by=[time_index] + self.dataloader.group_ids
#         ).reset_index(drop=True)
#         self.train_generators(data)
        
#         all_scores = np.full(
#             (
#                 data[data[time_index]<=time_range[-1]].shape[0], 
#                 len(features), self.pred_len
#             ), fill_value=np.nan, dtype=np.float32
#         )

#         for t in tqdm(time_range, disable=disable_progress):
#             # temporal slice [t- seq_len + 1:t + pred_len]
#             x = data[
#                 (data[time_index]  > t - self.seq_len) & 
#                 (data[time_index] <= (t + self.pred_len))
#             ]
            
#             _, data_loader = self.dataloader.create_timeseries(x)
#             y_t = self.model.predict(data_loader) 
            
#             self._local_attribute(x, t, y_t, all_scores)
        
#         # since we are trying to calculate the importance in future 
#         # feature importance are calculated from minimum time seq_len-1
#         # this drops those early nan values
#         num_group_ids = data[self.dataloader.group_ids].nunique().max()
#         all_scores = all_scores[num_group_ids*(self.seq_len-1):]
        
#         gc.collect()
#         return all_scores
        
#     @abc.abstractmethod
#     def get_name(self) -> str:
#         """
#         Return the name of the explainer.
#         """
#         return "Base"
        
# class FeatureOcclusion(BaseExplainer):
#     """
#     Feature Occlusion (FO) [1] computes attribution as the difference
#     in output after replacing each contiguous region with a given baseline. For time series we
#     considered continuous regions as features with in same time step or multiple time steps
#     grouped together.
    
#     [1] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In
#     European conference on computer vision, 2014.
#     """
#     def __init__(
#         self, model, dataloader:AgeData, 
#         features:List[str], n_samples:int=2, dists='unif'
#     ) -> None:
#         super().__init__(model, dataloader, features)
#         self.n_samples = n_samples
#         self.dists = dists
        
#     def train_generators(self, data):
#         """
#         Train and save any values necessary for later usage. 
#         This method is called before starting any attribute calculation.
#         """
#         self.minimums = data[self.features].min().values
#         self.maximums = data[self.features].max().values
        
#         self.means = data[self.features].mean().values
#         self.stds = data[self.features].std().values
        
#     def _local_attribute(self, x, t, y_t, all_scores):
#         selected_index = x[x[self.time_index] <= t].index
#         sample_size = selected_index.shape[0]
        
#         for feature_index, feature in enumerate(self.features):
#             x_hat = x.copy()
            
#             if self.dists == 'norm':
#                 x_hat.loc[selected_index, feature] = np.random.normal(
#                     self.means[feature_index], self.stds[feature_index], 
#                     size = sample_size
#                 )
#             else:
#                 x_hat.loc[selected_index, feature] = np.random.uniform(
#                     self.minimums[feature_index], self.maximums[feature_index], 
#                     size = sample_size
#                 )
            
#             _, dataloader = self.dataloader.create_timeseries(x_hat)
#             y_hat_t = self.model.predict(dataloader)
            
#             diff = abs(y_t[0] - y_hat_t[0]).detach().cpu().numpy()
            
#             # no need to return as arrays are passed by reference
#             all_scores[x[x[self.time_index] == t].index, feature_index, :] = diff
    
#     def get_name(self) -> str:
#         return "Feature Occlusion"
    
# class AugmentedFeatureOcclusion(BaseExplainer):
#     """
#     Augmented Feature Occlusion (AFO) [1], where we replace observations
#     with samples from the bootstrapped distribution p(xi) for each feature i. This is to avoid generating
#     out-of-distribution noise samples.
    
#     [1] Sana Tonekaboni, Shalmali Joshi, Kieran Campbell, David K Duvenaud, and Anna Goldenberg.
#     What went wrong and when? Instance-wise feature importance for time-series black-box models.
#     In Neural Information Processing Systems, 2020.
#     """
    
#     def __init__(
#         self, model, dataloader:AgeData, 
#         features:List[str], n_samples:int=2
#     ) -> None:
#         super().__init__(model, dataloader, features)
#         self.n_samples = n_samples
        
#     def train_generators(self, data):
#         self.dist = data[self.features].values
        
#     def _local_attribute(self, x, t, y_t, all_scores):
#         selected_index = x[x[self.time_index] <= t].index
#         sample_size = selected_index.shape[0]
        
#         for feature_index, feature in enumerate(self.features):
#             x_hat = x.copy()
#             x_hat.loc[selected_index, feature] = np.random.choice(
#                 self.dist[feature_index],
#                 size = sample_size
#             )
            
#             _, dataloader = self.dataloader.create_timeseries(x_hat)
#             y_hat_t = self.model.predict(dataloader)
            
#             diff = abs(y_t[0] - y_hat_t[0]).detach().cpu().numpy()
            
#             # no need to return as arrays are passed by reference
#             all_scores[x[x[self.time_index] == t].index, feature_index, :] = diff
    
#     def get_name(self) -> str:
#         return "Augmented Feature Occlusion"
    
# class FeatureAblation(BaseExplainer):
#     """
#         Feature Ablation (FA) [1] [2] computes attribution as the difference in
#         output after replacing each feature with a baseline. Input features can also be grouped
#         and ablated together rather than individually.
        
#         [1] Harini Suresh, Nathan Hunt, Alistair Johnson, Leo Anthony Celi, Peter Szolovits, and Marzyeh
#         Ghassemi. Clinical intervention prediction and understanding using deep networks. arXiv
#         preprint arXiv:1705.08498, 2017.
        
#         [2] Ozan Ozyegen, Igor Ilic, and Mucahit Cevik. Evaluation of interpretability methods for multivariate
#         time series forecasting. Applied Intelligence, pages 1–17, 2022.
#     """
#     def __init__(
#         self, model, dataloader:AgeData, 
#         features:List[str], method:str="global"
#     ) -> None:
#         """
#         Args:
#             method (str): Ablation method. Options: global, local. Defaults to "global".
#         """        """"""
#         super().__init__(model, dataloader, features)
        
#         assert method in ['global', 'local'], "Unknown method type"
#         self.method = method
        
#     def train_generators(self, data):
#         if self.method == "global":
#             self.baselines = data[self.features].mean().values
        
#     def _local_attribute(self, x, t, y_t, all_scores):
#         selected_index = x[x[self.time_index] <= t].index
        
#         for feature_index, feature in enumerate(self.features):
#             x_hat = x.copy()
            
#             if self.method == "global":
#                 x_hat.loc[selected_index, feature] = self.baselines[feature_index]
#             else:
#                 x_hat.loc[selected_index, feature] = x_hat[selected_index][feature].mean()
            
#             _, dataloader = self.dataloader.create_timeseries(x_hat)
#             y_hat_t = self.model.predict(dataloader)
            
#             diff = abs(y_t[0] - y_hat_t[0]).detach().cpu().numpy()
            
#             # no need to return as arrays are passed by reference
#             all_scores[x[x[self.time_index] == t].index, feature_index, :] = diff
    
#     def get_name(self) -> str:
#         return "Feature Ablation"
    

# class MorrisSensitivity(BaseExplainer):
#     """
#     Perform Morris Analysis [1] on model outputs using the mu_star.
    
#     API Reference: https://salib.readthedocs.io/en/latest/api.html#method-of-morris.
    
#     [1] Morris, M.D., 1991. Factorial Sampling Plans for Preliminary Computational 
#     Experiments. Technometrics 33, 161-174. https://doi.org/10.1080/00401706.1991.10484804
#     """
#     def __init__(
#         self, model, dataloader:AgeData, 
#         features:List[str], dists:[str | List[str]] = 'unif'
#     ) -> None:
#         """
#         Args:
#             dists [str | List[str]]: Sample distribution a feature. Defaults to 'unif'. 
#             Possible values are unif, norm, lognorm, triang.
#         """
#         super().__init__(model, dataloader, features)
        
#         if type(dists) == str:
#             self.dists = [dists] * len(features)
#         else:
#             assert len(features) == len(dists), \
#                 'Distribution list must equal to feature list in length'
#             self.dists = dists
        
#     def train_generators(self, data):
#         bounds = []
#         for i, feature in enumerate(self.features):
#             dist = self.dists[i]
#             values = data[feature]
#             if dist == 'unif':
#                 bounds.append([values.min(), values.max()])
#             elif dist == 'triang':
#                 bounds.append([values.min(), values.median(), values.max()])
#             else:
#                 bounds.append([values.mean(), values.std()])
        
#         self.sp = SALib.ProblemSpec({
#             "num_vars": len(self.features),
#             'names': self.features,
#             'bounds': bounds,
#             'dists': self.dists,
#             'sample_scaled': self.dataloader.scale # Whether the input samples are already scaled
#         })
#         self.num_group_ids = data[self.dataloader.group_ids].nunique().max()
        
#     def _local_attribute(self, x, t, y_t, all_scores):
#         selected_index = x[x[self.time_index] <= t].index
#         samples = SALib.sample.morris.sample(self.sp, self.num_group_ids)
#         samples_reshaped = samples.reshape((-1, self.num_group_ids, len(self.features)))
        
#         x_hat = x.copy()

#         num_morris_samples = samples_reshaped.shape[0]
#         y_hats = np.ndarray(
#             shape=(num_morris_samples, self.num_group_ids, self.dataloader.pred_len),
#             dtype=np.float32
#         )
        
#         for sample_index in range(num_morris_samples):
#             group_samples = samples_reshaped[sample_index]
#             group_samples = np.tile(group_samples, [self.dataloader.seq_len, 1])
            
#             x_hat.loc[selected_index, self.features] = group_samples
#             _, data_loader = self.dataloader.create_timeseries(x_hat)
            
#             y_hat_t = self.model.predict(data_loader)[0]
#             y_hats[sample_index] = y_hat_t

#         y_hats_reshaped = y_hats.reshape((-1, self.dataloader.pred_len))
#         for index, Y in enumerate(y_hats_reshaped.T):
#             morris_index = SALib.analyze.morris.analyze(
#                 self.sp, samples, Y
#             )['mu_star'].data
            
#             all_scores[
#                 x[x[self.dataloader.time_index] == t].index, :, index
#             ] = morris_index
    
#     def get_name(self) -> str:
#         return "Morris Sensitivity"
    
# def explainer_factory(
#     args, model, dataloader:AgeData, 
#     features: List[str | int]
# )-> BaseExplainer:
#     if args.explainer == 'FO':
#         explainer = FeatureOcclusion(model, dataloader, features, dists='unif')
#     elif args.explainer == 'AFO':
#         explainer = AugmentedFeatureOcclusion(model, dataloader, features, n_samples=2)
#     elif args.explainer == 'FA':
#         explainer = FeatureAblation(model, dataloader, features, method='global')
#     elif args.explainer == 'MS':
#         explainer = MorrisSensitivity(model, dataloader, features, dists='unif')
#     else:
#         raise ValueError(f'{args.explainer} isn\'t supported.')
    
#     print(f'Initialized explainer: {explainer.get_name()}.\n')
#     return explainer