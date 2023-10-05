import os, gc
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from typing import List

# from data.data_factory import AgeData

def add_day_time_features(dates):
    df_stamp = pd.DataFrame({'date': dates})

    # Time series library day encoding takes 3 values
    # check layers.Embed.TimeFeatureEmbedding class
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    
    return df_stamp.drop(columns=['date'])

def align_predictions(
    ground_truth:DataFrame, predictions_index:DataFrame, 
    predictions:List, dataloader,
    remove_negative:bool=True, upscale:bool=True
):
    horizons = range(dataloader.pred_len)
    time_index_max = predictions_index[dataloader.time_index].max()

    targets, time_index, group_ids = dataloader.targets, dataloader.time_index, dataloader.group_ids
    # a groupby with a groupength 1 throws warning later
    if len(group_ids) == 1: group_ids = group_ids[0]
    
    all_outputs = None 
    for target_index, target in enumerate(targets):
        if type(predictions[target_index]) == torch.Tensor:
            predictions[target_index] = predictions[target_index].numpy()

        pred_df = DataFrame(
            predictions[target_index], columns=horizons
        )
        pred_df = pd.concat([predictions_index, pred_df], axis=1)
        outputs = []

        for group_id, group_df in pred_df.groupby(group_ids):
            group_df = group_df.sort_values(
                by=time_index
            ).reset_index(drop=True)

            new_df = DataFrame({
                time_index : [t + time_index_max for t in range (1, dataloader.pred_len)]
            })
            new_df[group_ids] = group_id
            new_df.loc[:, horizons] = None
            new_df = new_df[group_df.columns]
            group_df = pd.concat([group_df, new_df], axis=0).reset_index(drop=True)

            for horizon in horizons:
                group_df[horizon] = group_df[horizon].shift(periods=horizon, axis=0)
                
            group_df[target] = group_df[horizons].mean(axis=1, skipna=True)
            outputs.append(group_df.drop(columns=horizons))

        outputs = pd.concat(outputs, axis=0)
        
        if all_outputs is None: all_outputs = outputs
        else: 
            all_outputs = all_outputs.merge(
                outputs, how='inner', on=predictions_index.columns
            )
            
    gc.collect()
      
    # upscale the target values if needed
    if upscale:
        all_outputs = dataloader.upscale_target(all_outputs)
        
    # must appear after upscaling
    if remove_negative:
        # remove negative values, since infection count can't be negative
        for target in targets:
            all_outputs.loc[all_outputs[target]<0, target] = 0
            
    # add `Predicted` prefix to the predictions
    all_outputs.rename(
        {target:'Predicted_'+target for target in targets}, 
        axis=1, inplace=True
    )
    
    # only keep the directly relevant columns
    ground_truth = ground_truth[[dataloader.date_index] + list(predictions_index.columns) + targets]
    # merge with grounth truth for evaluation
    all_outputs = ground_truth.merge(
        all_outputs, how='inner', on=list(predictions_index.columns)
    )
    
    return all_outputs

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr:0.5g}')


class EarlyStopping:
    def __init__(self, checkpoint_folder, patience=7, verbose=False, delta=0):
        self.checkpoint_folder = checkpoint_folder
        self.best_model_path = os.path.join(checkpoint_folder, 'checkpoint.pth')
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.5g} -> {val_loss:.5g}). Saving model ...')
            
        if os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder, exist_ok=True)
        
        # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        torch.save(model.state_dict(), self.best_model_path)
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)