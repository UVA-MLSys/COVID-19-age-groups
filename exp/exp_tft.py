# local classes and methods
from exp.config import DataConfig
from data.data_factory import AgeData
from utils.plotter import PlotResults
from utils.tools import align_predictions
from utils.metrics import calculate_metrics

# pytorch lightning
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

# pytorch forecasting
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE, MultiLoss
from pytorch_forecasting.data import TimeSeriesDataSet

import pandas as pd
import numpy as np
import gc, os
from datetime import datetime
    
class Experiment_TFT:
    def __init__(self, args, setting) -> None:
        self.args = args
        self.output_folder = args.result_path
        self.setting = setting
        self.output_folder = os.path.join(args.result_path, setting)
        
        if not os.path.exists(self.output_folder):
            print(f'Output folder {self.output_folder} does not exist. Creating ..')
            os.makedirs(self.output_folder, exist_ok=True)
        print(f'Starting experiment. Result folder {self.output_folder}.')
        
        self.progress_bar = not args.disable_progress
        
        self.age_dataloader = AgeData.build(args)
        self.plotter = PlotResults(
            self.output_folder, DataConfig.targets, 
            show=self.progress_bar
        )
        
    def _get_trainer(self):
        early_stop_callback = EarlyStopping(
            monitor="val_loss", verbose=True,  
            patience=self.args.patience
        )

        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html
        best_checkpoint = ModelCheckpoint(
            dirpath=self.output_folder, monitor="val_loss", 
            filename="best-{epoch}"
        )
        latest_checkpoint = ModelCheckpoint(
            dirpath=self.output_folder, every_n_epochs=1, 
            filename="latest-{epoch}"
        )

        # logging results to a tensorboard
        logger = TensorBoardLogger(self.output_folder)  

        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api
        trainer = Trainer(
            max_epochs = self.args.epochs,
            accelerator = 'auto',
            gradient_clip_val = self.args.clip,
            callbacks = [early_stop_callback, best_checkpoint, latest_checkpoint],
            logger = logger,
            enable_progress_bar = self.progress_bar,
            check_val_every_n_epoch = 1,
            enable_model_summary=False
            # deterministic=True
        )
        return trainer
    
    def _build_model(
        self, timeseries:TimeSeriesDataSet
    ):
        args = self.args
        # https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html
        tft = TemporalFusionTransformer.from_dataset(
            timeseries,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            attention_head_size=args.n_heads,
            lstm_layers=args.lstm_layers,
            learning_rate=args.learning_rate,
            loss=MultiLoss(
                [RMSE(reduction='mean') for _ in self.age_dataloader.targets]
            ), # Mean reduction turns this into MSE. For RMSE, reduction='sqrt-mean'
            optimizer='adam',
            log_interval=1,
            reduce_on_plateau_patience=1
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k.\n")
        return tft
        
    def train(
        self, train_data:pd.DataFrame, 
        val_data:pd.DataFrame, ckpt_path:str=None
    ):
        trainer = self._get_trainer()
        
        _, train_dataloader = self.age_dataloader.create_timeseries(
            data=train_data, train=True
        )
        val_timeseries, val_dataloader = self.age_dataloader.create_timeseries(data=val_data)
        tft = self._build_model(val_timeseries)
        
        start = datetime.now()
        print(f'\n----Training started at {start}----\n')
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=ckpt_path
        )
        
        end = datetime.now()
        print(f'\n----Training ended at {end}, elapsed time {end-start}.')
        print(f'Best model by validation loss saved at {trainer.checkpoint_callback.best_model_path}')
        
        best_model_path = trainer.checkpoint_callback.best_model_path
        print(f'Loading best model from {best_model_path}.')
        tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        
        return tft

    def test(
        self, tft:TemporalFusionTransformer, data:pd.DataFrame,
        split_type:str='Test', plot=True, output_results=True
    ) -> pd.DataFrame:
        _, dataloader = self.age_dataloader.create_timeseries(data)
        preds, x, test_index = tft.predict(
            dataloader, return_index=True, return_x=True,
            show_progress_bar=self.progress_bar
        )
        # target
        if output_results:
            self.output_results(
                trues=x['decoder_target'], preds=preds, 
                flag=split_type.lower()
            )

        result_merged = align_predictions(
            data, test_index, preds, self.age_dataloader
        )
        
        if plot:
            self.plotter.summed_plot(result_merged, type=split_type)
        gc.collect()
        return result_merged
    
    def output_results(
        self, trues:np.ndarray, preds:np.ndarray, 
        flag:str, filename='result.txt'
    ):
        # since we have single target only
        trues, preds = trues[0], preds[0]
        
        # upscale
        trues = self.age_dataloader.upscale_target(trues)
        preds = self.age_dataloader.upscale_target(preds)
        
        # remove neg values since cases can't be neg
        preds[preds<0] = 0
        trues[trues<0] = 0
        
        with open(filename, 'a') as output_file:
            evaluation_metrics = np.zeros(shape=4)
            mae, rmse, rmsle, r2 = calculate_metrics(preds, trues)
            result_string = f'{flag}: rmse:{rmse:0.5g}, mae:{mae:0.5g}, msle: {rmsle:0.5g}, r2: {r2:0.5g}'
            
            print(result_string)
            output_file.write(self.setting + ', ' + result_string + '\n')
            evaluation_metrics = [mae, rmse, rmsle, r2]
        
            np.savetxt(os.path.join(self.output_folder, f'{flag}_metrics.txt'), np.array(evaluation_metrics))
        
        np.save(os.path.join(self.output_folder, f'{flag}_pred.npy'), preds)
        np.save(os.path.join(self.output_folder, f'{flag}_true.npy'), trues)