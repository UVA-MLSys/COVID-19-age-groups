import os, torch
from tqdm import tqdm 
import pandas as pd
from utils.interpreter import *
from tint.metrics import mae, mse
from datetime import datetime
from captum.attr import (
    DeepLift,
    GradientShap,
    IntegratedGradients,
    Lime,
    FeaturePermutation
)

from tint.attr import (
    AugmentedOcclusion,
    Occlusion, 
    FeatureAblation
)

expl_metric_map = {
    'mae': mae, 'mse': mse
}

explainer_name_map = {
    "deep_lift":DeepLift,
    "gradient_shap":GradientShap,
    "integrated_gradients":IntegratedGradients,
    "lime":Lime, # very slow
    "occlusion":Occlusion,
    "augmented_occlusion":AugmentedOcclusion, # requires data when initializing
    "feature_ablation":FeatureAblation,
    "feature_permutation":FeaturePermutation,
}

class Exp_Interpret:
    def __init__(
        self, exp, dataloader
    ) -> None:
        assert not exp.args.output_attention, 'Model needs to output target only'
        self.exp = exp
        self.args = exp.args
        
        self.result_folder =  os.path.join(exp.output_folder, 'interpretation') 
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder, exist_ok=True)
        print(f'Interpretations will be saved in {self.result_folder}')
        
        self.device = exp.device
        exp.model.eval().zero_grad()
        self.model = exp.model
        
        self.explainers_map = dict()
        for name in exp.args.explainers:
            if name in ['augmented_occlusion']:
                all_inputs = get_total_data(dataloader, self.device)
                
                self.explainers_map[name] = explainer_name_map[name](
                    self.model, data=all_inputs
                )
            else:
                explainer = explainer_name_map[name](self.model)
                self.explainers_map[name] = explainer
                
    def run_regressor(self, dataloader, name):
        results = [['batch_index', 'metric', 'tau', 'area', 'comp', 'suff']]
        progress_bar = tqdm(
            enumerate(dataloader), total=len(dataloader), 
            disable=self.args.disable_progress
        )
        attr = []
        
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in progress_bar:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
            
            inputs = (batch_x, batch_x_mark)
            # baseline must be a scaler or tuple of tensors with same dimension as input
            baselines = get_baseline(inputs, mode=self.args.baseline_mode)
            additional_forward_args = (dec_inp, batch_y_mark)

            # get attributions
            batch_attr, batch_results = self.evaluate_regressor(
                name, inputs, baselines, 
                additional_forward_args, batch_index
            )
            results.extend(batch_results)
            attr.append(batch_attr)
            
        attr = tuple(torch.vstack([a[i] for a in attr]) for i in range(2))
        return attr, results
    
    def interpret(self, dataloader):
        for name in self.args.explainers:
            results = []
            start = datetime.now()
            print(f'Running {name} from {start}')
            
            attr, results = self.run_regressor(dataloader, name)
            
            end = datetime.now()
            print(f'Experiment ended at {end}. Total time taken {end - start}.')
            self.dump_results(results, f'{name}.csv')
            
            attr_output_file = f'{self.args.flag}_{name}.pt' 
            attr_output_path = os.path.join(self.result_folder, attr_output_file)
            
            attr_numpy = tuple([a.detach().cpu().numpy() for a in attr])
            torch.save(attr_numpy, attr_output_path)
                
    def evaluate_regressor(
        self, name, inputs, baselines, 
        additional_forward_args, batch_index
    ):
        explainer = self.explainers_map[name]
        
        attr = compute_regressor_attr(
            inputs, baselines, explainer, 
            additional_forward_args, self.args, 
        )
    
        results = []
        # get scores
        for metric_name in ['mae', 'mse']:
            # batch x pred_len x seq_len x features
            for tau in range(self.args.pred_len):
                if type(attr) == tuple:
                    attr_per_pred = tuple([
                        attr_[:, tau] for attr_ in attr
                    ])
                else: attr_per_pred = attr[:, tau]
                
                for area in self.args.areas:
                    metric = expl_metric_map[metric_name]
                    error_comp = metric(
                        self.model, inputs=inputs, 
                        attributions=attr_per_pred, baselines=baselines, 
                        additional_forward_args=additional_forward_args,
                        topk=area, mask_largest=True
                    )
                    
                    error_suff = metric(
                        self.model, inputs=inputs,
                        attributions=attr_per_pred, baselines=baselines, 
                        additional_forward_args=additional_forward_args,
                        topk=area, mask_largest=False
                    )
            
                    result_row = [
                        batch_index, metric_name, tau, area, error_comp, error_suff
                    ]
                    results.append(result_row)
    
        return attr, results
        
    def dump_results(self, results, filename):
        results_df = pd.DataFrame(results[1:], columns=results[0])
        
        batch_filename = os.path.join(self.result_folder, f'batch_{filename}')
        results_df.round(6).to_csv(batch_filename, index=False)
        
        results_df = results_df.groupby(['metric', 'area'])[
            ['comp', 'suff']
        ].aggregate('mean').reset_index()
        
        filepath = os.path.join(self.result_folder, filename)
        results_df.round(6).to_csv(filepath, index=False)
        print(results_df)