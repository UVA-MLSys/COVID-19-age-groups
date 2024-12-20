from pandas import to_datetime, to_timedelta
from typing import Optional, Callable
from dataclasses import dataclass

@dataclass
class DataConfig:
    root_folder = './dataset/processed'
    
    static_reals=["UNDER5","AGE517","AGE1829","AGE3039","AGE4049","AGE5064","AGE6574","AGE75PLUS"]
    observed_reals=['VaccinationFull', 'Cases']
    targets=['Cases']
    
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
        self, args, train_start, val_start, test_start, 
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
            
        if args.percent < 100:
            days = (self.val_start - self.train_start).days - args.seq_len - args.pred_len
            assert days >= 0, "not enough data to split using percent"
            self.train_start = self.train_start + to_timedelta(days * (100 - args.percent) // 100, unit='day')

            print(f'Using {days * args.percent // 100} days of training data.')
            
        print(f'Train start: {self.train_start}, val start: {self.val_start}, test start: {self.test_start}, test end: {self.test_end}')
        
            
    @staticmethod
    def primary(args):
        return Split(
            args=args, train_start="2020-03-01", val_start="2021-11-28",
            test_start="2021-12-12", test_end="2021-12-25",
            transform=to_datetime
        )