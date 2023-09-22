# local classes and methods
from exp.config import DataConfig, ModelConfig
from data.dataloader import AgeData
from utils.plotter import PlotResults
from utils.utils import align_predictions

# pytorch lightning
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

# pytorch forecasting
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE, MultiLoss
from pytorch_forecasting.data import TimeSeriesDataSet

import pandas as pd
import gc
from datetime import datetime
    
class Experiment_TFT:
    def __init__(
        self, data_path:str, result_folder:str,
        progress_bar:bool=True
    ) -> None:
        self.result_folder = result_folder
        self.progress_bar = progress_bar
        
        self.age_dataloader = AgeData(
            data_path=data_path,
            date_index=DataConfig.date_index, 
            seq_len=DataConfig.seq_len, pred_len=DataConfig.pred_len,
            group_ids=DataConfig.group_ids, 
            static_reals=DataConfig.static_reals,
            observed_reals=DataConfig.observed_reals,
            known_reals=DataConfig.known_reals,
            targets=DataConfig.targets,
            scale=DataConfig.scale
        )
        self.model_config = ModelConfig.primary()
        self.plotter = PlotResults(
            result_folder, DataConfig.targets, 
            show=self.progress_bar
        )
        
    def _get_trainer(self):
        early_stop_callback = EarlyStopping(
            monitor="val_loss", verbose=True,  
            patience=self.model_config.early_stopping_patience
        )

        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html
        best_checkpoint = ModelCheckpoint(
            dirpath=self.result_folder, monitor="val_loss", filename="best-{epoch}"
        )
        latest_checkpoint = ModelCheckpoint(
            dirpath=self.result_folder, every_n_epochs=1, filename="latest-{epoch}"
        )

        # logging results to a tensorboard
        logger = TensorBoardLogger(self.result_folder)  

        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api
        trainer = Trainer(
            max_epochs = self.model_config.epochs,
            accelerator = 'auto',
            gradient_clip_val = self.model_config.gradient_clip_val,
            callbacks = [early_stop_callback, best_checkpoint, latest_checkpoint],
            logger = logger,
            enable_progress_bar = self.progress_bar,
            check_val_every_n_epoch = 1,
            enable_model_summary=False,
            # deterministic=True
        )
        return trainer
    
    def _build_model(
        self, parameters:dict,
        timeseries:TimeSeriesDataSet
    ):
        print(f'\nModel configuration {parameters}.')
        
        # https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html
        tft = TemporalFusionTransformer.from_dataset(
            timeseries, **parameters,
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
        self, model_config:ModelConfig, train_data:pd.DataFrame, 
        val_data:pd.DataFrame, ckpt_path:str=None
    ):
        trainer = self._get_trainer()
        
        _, train_dataloader = self.age_dataloader.create_timeseries(
            data=train_data, train=True
        )
        val_timeseries, val_dataloader = self.age_dataloader.create_timeseries(data=val_data)
        tft = self._build_model(
            model_config.model_parameters, val_timeseries
        )
        
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
        print(f'Loading best model from {best_model_path}.\n\n')
        tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        
        return tft

    def test(
        self, tft:TemporalFusionTransformer, data:pd.DataFrame,
        split_type:str='Test', plot=True
    ) -> pd.DataFrame:
        _, dataloader = self.age_dataloader.create_timeseries(data)
        predictions, test_index = tft.predict(
            dataloader, return_index=True, 
            show_progress_bar=self.progress_bar
        )

        result_merged = align_predictions(
            data, test_index, predictions, self.age_dataloader
        )
        if plot:
            self.plotter.summed_plot(result_merged, type=split_type)
        gc.collect()
        return result_merged    