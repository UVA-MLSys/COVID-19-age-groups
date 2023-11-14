import pandas as pd
from exp.exp_interpret import explainer_name_map

explainers = [
    'feature_ablation', 'feature_permutation','morris_sensitivity',
    'occlusion','augmented_occlusion',
    'deep_lift','integrated_gradients', 'gradient_shap'
    
]
model_name = 'FEDformer'


# for name in explainers:
#     filename = f'results/FEDformer_Total/interpretation/batch_{name}.csv'
#     df = pd.read_csv(filename)
#     mean = df.groupby('metric')[['comp', 'suff']].aggregate('mean').reset_index()
#     print(name, '\n', mean, '\n')

for name in explainers:
    full_name = explainer_name_map[name].get_name()
    filename =  f'results/FEDformer_Total/interpretation/test_int_metrics_{full_name}.csv'
    df = pd.read_csv(filename)
    MAE = df[df['metrics']=='normalized_mae']['values'].values[0]
    RMSE = df[df['metrics']=='normalized_rmse']['values'].values[0]
    NDCG = df[df['metrics']=='normalized_ndcg']['values'].values[0]
    print(f'{full_name} & {MAE:0.4f} & {RMSE:0.4f} & {NDCG:0.4f} \\\\')