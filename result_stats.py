import pandas as pd
explainers = [
    'deep_lift','gradient_shap','integrated_gradients',
    'lime','occlusion','augmented_occlusion','feature_ablation',
    'feature_permutation','morris_sensitivity'
]
model_name = 'FEDformer'


for name in explainers:
    filename = f'results/FEDformer_Total/interpretation/batch_{name}.csv'
    df = pd.read_csv(filename)
    mean = df.groupby('metric')[['comp', 'suff']].aggregate('mean').reset_index()
    print(name, '\n', mean, '\n')