import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
warnings.filterwarnings('ignore')


class Dataset_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',scale=True, timeenc=0, freq='h'):

        self.args = args
        # info
        if size == None:
            self.seq_len = 7 * 4 * 4
            self.label_len = 7 * 4
            self.pred_len = 7 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        num_train = df_raw[(df_raw['Datetime'] >= '2023-02-01') & (df_raw['Datetime'] <= '2024-06-31')].shape[0]
        num_vali =  df_raw[(df_raw['Datetime'] >= '2024-07-01') & (df_raw['Datetime'] <= '2024-09-31')].shape[0]
        num_test =  df_raw[(df_raw['Datetime'] >= '2024-10-01')].shape[0] 
        
        border1s = [0,
                    num_train  -seq_len,
                    len(df_raw) - num_test - seq_len]

        border2s = [num_train,
                    num_train  + num_vali,
                    len(df_raw)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['Datetime']][border1:border2]
        df_stamp['Datetime'] = pd.to_datetime(df_stamp.Datetime)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Datetime.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Datetime.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Datetime.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.Datetime.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['Datetime'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Datetime'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
