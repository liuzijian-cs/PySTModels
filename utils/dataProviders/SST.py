from utils.dataProviders.BasicDataProvider import BasicDataProvider
import pandas as pd
import torch
import numpy as np
import os
from torch import Tensor


class DataProvider(BasicDataProvider):
    def __init__(self, args, scaler=None, device=None):
        super(DataProvider, self).__init__(args, scaler, device)

    def _load_data(self):
        df = pd.read_csv(self.data_path, index_col=0)
        return df.iloc[1:,:10].to_numpy()

    def _split_data(self, data):
        """
        Split the dataset strictly in time order
        :return data_train, data_valid, data_test
        """
        # Train-Valid-Test Split
        train_ratio = 1 - self.valid_ratio - self.test_ratio
        if train_ratio < 0 or train_ratio > 1:
            raise ValueError('Invalid ratio settings. The sum of train, valid and test ratio must be 1')
        train_end = int(train_ratio * data.shape[0])
        valid_end = train_end + int(self.valid_ratio * data.shape[0])
        return data[:train_end], data[train_end:valid_end], data[valid_end:]

    def _prepare_data(self, data_type) -> tuple[Tensor, Tensor]:
        """
        :return: x :torch.Tensor [T, seq_len, N] , y :torch.Tensor [T, pred_len, N]
        """
        assert data_type in ['train', 'valid', 'test', 'all']
        end_index = len(self.data_dict[data_type]) - self.seq_len - self.pred_len + 1
        x_shape = (end_index, self.seq_len, self.data_dict[data_type].shape[-1])
        y_shape = (end_index, self.pred_len, self.data_dict[data_type].shape[-1])
        # Preallocate memory
        x = torch.zeros(x_shape, dtype=torch.float, device=self.device)
        y = torch.zeros(y_shape, dtype=torch.float, device=self.device)
        for i in range(end_index):
            x[i] = torch.tensor(self.data_dict[data_type][i:i + self.seq_len], device=self.device)
            y[i] = torch.tensor(
                self.data_dict[data_type][i + self.seq_len: i + self.seq_len + self.pred_len],
                device=self.device)
        return x, y