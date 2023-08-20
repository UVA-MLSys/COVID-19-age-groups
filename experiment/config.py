from pandas import to_datetime
from typing import Optional, Callable
from dataclasses import dataclass

@dataclass
class DataConfig:
    root_folder = 'dataset/processed'
    date_index='Date'
    seq_len=13 
    pred_len=15
    
    static_reals=["UNDER5","AGE517","AGE1829","AGE3039","AGE4049","AGE5064","AGE6574","AGE75PLUS"]
    observed_reals=['VaccinationFull', 'Cases']
    known_reals=["SinWeekly"]
    targets=['Cases']
    
    group_ids=['FIPS']
    scale=True
    
@dataclass
class FeatureFiles:
    root_folder = 'dataset/raw'
    population_cut = [20, 100, 500]
    
    # feature files and corresponding feature columns 
    static_features = {
        "Age Groups.csv": ["UNDER5", "AGE517","AGE1829","AGE3039","AGE4049","AGE5064","AGE6574","AGE75PLUS"]
    }
    dynamic_features = {
        "Vaccination.csv": "VaccinationFull"
    }
    targets = {
        "Cases.csv": "Cases"
    } 
    
    # support file
    population = "Population.csv"
    
    # data preprocessing
    remove_input_outliers = True
    remove_target_outliers = True
    first_date = to_datetime('2020-03-01')
    last_date = to_datetime('2022-12-31')

class Split:
    def __init__(
        self, train_start, val_start, test_start, 
        test_end, transform: Optional[Callable]=None 
    ):
        self.train_start = train_start
        self.val_start = val_start
        self.test_start = test_start
        self.test_end = test_end

        if transform:
            self.train_start = transform(train_start)
            self.val_start = transform(val_start)
            self.test_start = transform(test_start)
            self.test_end = transform(test_end)
            
    @staticmethod
    def primary():
        return Split(
            train_start="2020-03-01", val_start="2021-11-28",
            test_start="2021-12-13", test_end="2021-12-27",
            transform=to_datetime
        )
    
    @staticmethod
    def updated():
        return Split(
            train_start="2022-01-01", val_start="2022-12-02",
            test_start="2022-12-17", test_end="2022-12-31",
            transform=to_datetime
        )
             
class ModelConfig:
    def __init__(
        self, 
        model_parameters:dict = {
            "hidden_size":16,
            "dropout": 0.1,
            "attention_head_size": 4,
            "learning_rate": 1e-3,
            "lstm_layers": 1
        },
        early_stopping_patience:int = 3,
        epochs:int = 10,
        gradient_clip_val: Optional[float] = 1
    ) -> None:
        self.model_parameters = model_parameters
        self.early_stopping_patience = early_stopping_patience
        self.epochs = epochs
        self.gradient_clip_val = gradient_clip_val
        
    @staticmethod
    def primary():
        return ModelConfig()