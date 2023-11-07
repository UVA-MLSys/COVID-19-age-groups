from utils.utils import add_day_time_features
import pandas as pd, numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class MultiTimeSeries(Dataset):
    def __init__(
        self, data, seq_len, pred_len, past_features,
        known_features, targets, time_col='Date', 
        id_col='FIPS', 
    ):
    
        self.seq_len = seq_len
        self.label_len = seq_len // 2
        self.pred_len = pred_len
        
        self.past_features = past_features
        self.known_features = known_features
        self.targets = targets
        
        self.id_col = id_col
        self.time_col = time_col
        self.time_steps = self.seq_len + self.pred_len

        self.__load_data__(data)
        
    def __load_data__(self, df_raw):
        df_raw[self.time_col] = pd.to_datetime(df_raw[self.time_col])
        
        id_col, time_steps = self.id_col, self.time_steps
        df_raw = df_raw.sort_values(by=[self.time_col, id_col]).reset_index(drop=True)
        
        data_stamp = add_day_time_features(df_raw.loc[0, [self.time_col]].values)
        time_encoded_columns = list(data_stamp.columns)
        print('Time encoded columns :', time_encoded_columns)
            
        print('Getting valid sampling locations.')
        
        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in df_raw.groupby(id_col):
            num_entries = len(df)
            if num_entries >= time_steps:
                valid_sampling_locations += [
                    (identifier, i)
                    for i in range(num_entries - time_steps + 1)
                ]
            split_data_map[identifier] = df
        
        max_samples = len(valid_sampling_locations)
        self.ranges = valid_sampling_locations
        
        # must add target features to the end of data. this is the input to encoder decoder 
        past_features = [f for f in self.past_features if f not in self.targets] + self.targets
        self.data = np.zeros((max_samples, self.time_steps, len(past_features)))
        self.data_stamp = np.zeros((max_samples, self.time_steps, len(time_encoded_columns)))
        self.target_data = np.zeros((max_samples, self.time_steps, len(self.targets)))
        
        for i, tup in tqdm(enumerate(valid_sampling_locations), mininterval=300):
            # if ((i + 1) % 10000) == 0:
            #     print(i + 1, 'of', max_samples, 'samples done...')
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx:start_idx + time_steps]
            self.data[i] = sliced[self.past_features]
            self.data_stamp[i] = add_day_time_features(sliced[self.time_col].values)
            self.target_data[i] = sliced[self.targets]
        
    def __getitem__(self, index):
        s_end = self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[index][:s_end]
        seq_y = self.data[index][r_begin:r_end]
        
        seq_x_mark = self.data_stamp[index][:s_end]
        seq_y_mark = self.data_stamp[index][r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) # - self.seq_len - self.pred_len + 1