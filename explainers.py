import numpy as np
from typing import List, Union
import SALib, torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

class MorrisSensitivty:
    def __init__(self, model, data, pred_len) -> None:
        self.model = model
        self.pred_len = pred_len
        
        self.sp = self._build_problem_spec(data)
        
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
            'names':list(range(n_features))
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