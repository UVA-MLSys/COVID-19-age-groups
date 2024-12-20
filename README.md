# Introduction

12/26/24: Our paper titled **[Interpreting Time Series Transformer Models and Sensitivity Analysis of Population Age Groups to COVID-19 Infections](https://arxiv.org/abs/2401.15119)** has been accepted at the AAAI 2024 workshop [AI4TS: AI FOR TIME SERIES ANALYSIS](https://ai4ts.github.io/aaai2024#introduction)

This study aimed to identify the most influential age groups in COVID-19 infection rates at the US county level using the time series interpretation techniques with deep learning. Our approach involved training the state-of-the-art time-series models on different age groups as a static feature and the population vaccination status as the dynamic feature. We analyzed the impact of those age groups on COVID-19 infection rates by perturbing individual input features and ranked them based on their sensitivity scores. The findings are verified using ground truth data from the CDC and US Census, which provide the true infection rates for each age group. The results suggest that young adults were the most influential age group in COVID-19 transmission at the county level between March 1, 2020, and November 27, 2021. Using these results can inform public health policies and interventions, such as targeted vaccination strategies, to better control the spread of the virus. 

## Citation

```
@article{islam2024interpreting,
  title={Interpreting Time Series Transformer Models and Sensitivity Analysis of Population Age Groups to COVID-19 Infections},
  author={Islam, Md Khairul and Valentine, Tyler and Sue, Timothy Joowon and Karmacharya, Ayush and Benham, Luke Neil and Wang, Zhengguang and Kim, Kingsley and Fox, Judy},
  journal={arXiv preprint arXiv:2401.15119},
  year={2024}
}
```

## Folder Structure
* `data`: data loader and merger files

* `dataset`: raw and processed data set in CSV files. 
  * The `Total.csv` file isn't committed here. You can download from [here](https://drive.google.com/file/d/1Osw2csQ_tb5i9eGIQAWYMJEM29lZ8tWF/view?usp=sharing), unzip the files and keep them in the same path.  
  * The cached datasets (`.pt` files) are also saved here. Some initial execution times can be save by downloading them from the [drive]((https://drive.google.com/drive/folders/1IPID82QWKUTxGOrynLn_QhYgVIa9CjUq?usp=sharing)) and keeping in the same path. The code will automatically recognise the caches and load from there instead of rebuilding. Make sure to remove the cache if you changed some data config (e.g. `seq_len`, `pred_len`), so that they are rebuild.

* `exp`: experiment runner and configuration for data, model, plots.

* `layers`: neural network layer classes and related utils.

* `models`: timeseries model classes.

* `results`: result output from the model training and testing.

* `scratch`: folder to run temporary experiments in without git tracking.

* `scripts`: template scripts and slurm job scripts for rivanna and cs remote server.

* `utils`: miscellaneous util methods and result plotter.

* `singilarity.def`: definition file for singularity.

## Features

The following table lists the features with their source and description. Note that, past values of the target and known futures are also used as observed inputs.

<div align="center" style="overflow-x:auto;text-align:center;vertical-align: middle;">
<table border="1">
<caption> <h2>Details of Features </h2> </caption>
<thead>
<tr>
<th>Feature</th>
<th>Type</th>
<th>Update Frequency</th>
<th>Description</th>
<th>Source(s)</th>
</tr>

</thead>

<tbody>

<tr>
<td><strong>Age Groups</strong> <br>( UNDER5, AGE517, AGE1829, AGE3039, AGE4049, AGE5064, AGE6574, AGE75PLUS )</td>
<td>Static</td>
<td>Once</td>
<td>Percent of population in each age group.</td>
<td><span><a href="https://www.census.gov/data/tables/time-series/demo/popest/2020s-national-detail.html" target="_blank">2020 Govt Census</a></span></td>
</tr>

<tr>
<td><strong>Vaccination Full Dose</strong> <br>(Series_Complete_Pop_Pct)</td>
<td>Observed</td>
<td rowspan=3>Daily</td>
<td> Percent of people who are fully vaccinated (have second dose of a two-dose vaccine or one dose of a single-dose vaccine) based on the jurisdiction and county where recipient lives.</td>
<td><span><a href="https://www.unacast.com/covid19/social-distancing-scoreboard" target="_blank">CDC</a></span></td>
</tr>

<tr>
<td><strong>Time encoded features</strong></td>
<td>Known Future</td>
<td> <em> Features calculated from time </em>.</td>
<td>Date</td>
</tr>

<tr>
<td><strong>Case</strong></td>
<td>Target</td>
<td> COVID-19 infection at county level.</td>
<td><span><a href="https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/" target="_blank">USA Facts</a></span></td>
</tr>

</tbody>
</table>

</div>

## Models

* DLinear
* Autoformer
* FEDformer
* PatchTST
* TimesNet
* Transformer

## Interpretation Techniques

* Feature Ablation
* Feature Occlusion
* Augmented Feature Occlusion
* Deep Lift
* Integrated Gradients
* Gradient Shap
* Morris Sensitivity 
* Lime 
* Feature Permutation

## Results

### Test Results

The following table shows the test results for the COVID-19 dataset, calculated daily at each US county.
<table>
 <tr>
  <th>Model</th>
  <th>mae</th>
  <th>rmse</th>
  <th>rmsle</th>
  <th>r2</th>
 </tr>
 <tr>
  <td>Autoformer</td>
  <td>33.701</td>
  <td>188.330</td>
  <td>1.826</td>
  <td>0.457</td>
 </tr>
 <tr>
  <td>DLinear</td>
  <td>29.525</td>
  <td>174.950</td>
  <td>1.425</td>
  <td>0.524</td>
 </tr>
 <tr>
  <td>FEDformer</td>
  <td>31.490</td>
  <td>180.450</td>
  <td>1.659</td>
  <td>0.499</td>
 </tr>
 <tr>
  <td>PatchTST</td>
  <td>33.174</td>
  <td>183.647</td>
  <td>1.530</td>
  <td>0.469</td>
 </tr> 
 <tr>
  <td>TimesNet</td>
  <td>34.354</td>
  <td>191.920</td>
  <td>1.604</td>
  <td>0.415</td>
 </tr>
 <tr>
  <td>Transformer</td>
  <td>34.818</td>
  <td>192.720</td>
  <td>1.601</td>
  <td>0.410</td>
</tr> 
</table>

AOPCR results of the interpretation using the FEDformer model on the test set. Comp. and Suff. are the short form of comprehensiveness and sufficiency.

| Interpretation Method | Comp. (MAE) | Comp. (MSE) |Suff. (MAE) | Suff. (MSE) |
|:---|:---|:---|:---|:---|
| Feature Ablation | 4.91 | 8.64 | 9.53 | 10.50 |
| Feature Permutation | 4.00 | 7.08 | 8.00 | 8.28 | 
| Feature Occlusion | 4.89 | 8.44 | 9.49 | 10.40 |
| Augmented Feature Occlusion | 4.18 | 7.66 | 7.96 | 8.09 |
| Deep Lift | 5.72 | 9.54 | 8.90 | 9.43 |
| Gradient Shap | 4.78 | 8.17 | 8.04 | 8.27 |
| Integrated Gradients | 5.52 | 9.09 | 9.25 | 10.20 |


## Setup Environment

### 1. Singularity (Recommended on Rivanna)
#### Option A. Pull already built container

Pull the singularity container from the remote library,
```bash
singularity pull timeseries.sif library://khairulislam/collection/timeseries:latest
```

#### Option B. Build container from scratch

*Note:* If you want to create the container from scrach, you need a linux machine with `root` privilege or build remotely at [cloud.sylabs.io/library](https://cloud.sylabs.io/library). On Rivanna you can't create containers, you are not the `root`. The you can use the [singularity.def](/singularity.def) file. After compilation, you'll get a container named `timeseries.sif`. 

```bash
sudo singularity build timeseries.sif singularity.def
```

#### Running scripts in container
Then you can use the container to run the scripts. `--nv` indicates using GPU. For example, 
```bash
singularity run --nv timeseries.sif python run.py
```

### 2. Virtual Environment  (More flexible compared to container)
If you are on remote servers like Rivanna or UVA CS server, you don't have the permission to upgrade default python version. But you can use the already installed `Anaconda` to create a new environment and install latest python and libraries there. 

To create a new env with name `ml` and python version `3.10` run the following. `python 3.10` is the latest one supported with `pytorch GPU` at this moment. 
```bash
conda create --name ml python=3.10

# activates the virtual env
conda activate ml

# installs libraries in the default pip and this env
pip install some_library 

# installs libraries in this env, but often slower and doesn't work
conda install some_library 

# will run the script with this environment
python run.py

# deactivates the virtual env
conda deactivate
```

You should install pytorch with cuda separately, since pip server can't find it

```bash
pip install torch==1.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Then install the other libraries from [requirement.txt](/requirements.txt). If you have trouble installing from the file (command `pip install -r requirements.txt`), try installing each library manually. 

### 3. GPU 
Next, you might face issues getting GPU running on Rivanna. Even on a GPU server the code might not recognize the GPU hardware if cuda and cudnn are not properly setup. Try to log into an interactive session in a GPU server, then run the following command from terminal,

```bash
python -c "import torch;print(torch.cuda.is_available())"
```

If this is still 0, then you'll have to install the cuda and cudnn versions that match version in `nvidia-smi` command output. Also see if you tensorflow version is for CPU or GPU. For this project, `tensorflow` isn't used. So no need to install it.

## Reproduce

### Running models
Use the `run.py` to run the available models on the dataset. See `scripts/commands.sh` file for some examples. All commands must be run from this current folder. Not from any sub-folders. We currently support the following models `DLinear, Autoformer, FEDformer, PatchTST, TimesNet` from the[Time-Series-Library](https://github.com/thuml/Time-Series-Library). Note that, anything written in the `scratch` folder will be ignored by `git`, since it is added in `.gitignore`. So setting `--result_path scratch` is a good idea for temporary experiments.

```
$COVID-19-age-groups> python run.py --help

Run Timeseries Models

options:
  -h, --help            show this help message and exit
  --test                test the checkpointed best model, train otherwise (default: False)
  --model {DLinear,Autoformer,FEDformer,PatchTST,TimesNet}
                        model name (default: DLinear)
  --seed SEED           random seed (default: 7)
  --root_path ROOT_PATH
                        root path of the data file (default: ./dataset/processed/)
  --data_path DATA_PATH
                        data file (default: Top_20.csv)
  --result_path RESULT_PATH
                        result folder (default: results)
  --freq {s,t,h,d,b,w,m}
                        freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min
                        or 3h (default: d)
  --no-scale            do not scale the dataset (default: False)
  --seq_len SEQ_LEN     input sequence length (default: 14)
  --label_len LABEL_LEN
                        start token length (default: 7)
  --pred_len PRED_LEN   prediction sequence length (default: 14)
  --top_k TOP_K         for TimesBlock (default: 5)
  --num_kernels NUM_KERNELS
                        for Inception (default: 6)
  --d_model D_MODEL     dimension of model (default: 64)
  --n_heads N_HEADS     num of heads (default: 4)
  --e_layers E_LAYERS   num of encoder layers (default: 2)
  --d_layers D_LAYERS   num of decoder layers (default: 1)
  --d_ff D_FF           dimension of fcn (default: 256)
  --moving_avg MOVING_AVG
                        window size of moving average (default: 7)
  --factor FACTOR       attn factor (default: 3)
  --distil              whether to use distilling in encoder, using this argument means not using distilling (default: True)
  --dropout DROPOUT     dropout (default: 0.1)
  --embed {timeF,fixed,learned}
                        time features encoding (default: timeF)
  --activation ACTIVATION
                        activation (default: gelu)
  --output_attention    whether to output attention in ecoder (default: False)
  --num_workers NUM_WORKERS
                        data loader num workers (default: 0)
  --train_epochs TRAIN_EPOCHS
                        train epochs (default: 10)
  --batch_size BATCH_SIZE
                        batch size of train input data (default: 32)
  --patience PATIENCE   early stopping patience (default: 3)
  --learning_rate LEARNING_RATE
                        optimizer learning rate (default: 0.001)
  --des DES             exp description (default: )
  --loss LOSS           loss function (default: MSE)
  --lradj {type1,type2}
                        adjust learning rate (default: type1)
  --use_amp             use automatic mixed precision training (default: False)
  --no_gpu              do not use gpu (default: False)
  --gpu GPU             gpu (default: 0)
  --use_multi_gpu       use multiple gpus (default: False)
  --devices DEVICES     device ids of multile gpus (default: 0,1,2,3)
  --p_hidden_dims P_HIDDEN_DIMS [P_HIDDEN_DIMS ...]
                        hidden layer dimensions of projector (List) (default: [64, 64])
  --p_hidden_layers P_HIDDEN_LAYERS
                        number of hidden layers in projector (default: 2)
  --disable_progress    disable progress bar (default: False)
```

### Interpreting models

```
$COVID-19-age-groups> python interpret_with_ground_truth.py --help

Interpret Timeseries Models

options:
 same as before, additional arguments

  --explainers [{deep_lift,gradient_shap,integrated_gradients,lime,occlusion,augmented_occlusion,feature_ablation,feature_permutation,morris_sensitivity} ...]
                        explaination method names (default: ['feature_ablation'])
  --flag {train,val,test,updated}
    flag for data split (default: test)
  --baseline_mode {random,aug,zero,mean}
    how to create the baselines for the interepretation methods (default: random)
```

```
$COVID-19-age-groups> python interpret_without_ground_truth.py --help

options:
same as before, additional arguments

--metrics [METRICS ...]
    interpretation evaluation metrics (default: ['mae', 'mse'])
--areas [AREAS ...]   top k features to keep or mask during evaluation (default: [0.05, 0.1])

```


### Submitting job scripts

To submit job scripts in remote servers, use the templates in the `scripts` folder. And submit the jobs from this current folder.
```
$COVID-19-age-groups> sbatch scripts/rivanna_slurm.sh
```
The job outputs will be saved in [`scripts/outputs`](/scripts/outputs/) folder. The model outputs will be in the `result_path/setting` folder.


## Usage guideline

* Please do not add temporarily generated files in this repository.
* Make sure to clean your tmp files before pushing any commits.
* In the .gitignore file you will find some paths in this directory are excluded from git tracking. So if you create anything in those folders, they won't be tracked by git. 
  * To check which files git says untracked `git status -u`. 
  * If you have folders you want to exclude add the path in `.gitignore`, then `git add .gitignore`. Check again with `git status -u` if it is still being tracked.
