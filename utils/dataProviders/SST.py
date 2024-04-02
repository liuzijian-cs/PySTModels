from utils.dataProviders.BasicDataProvider import BasicDataProvider
import pandas as pd
import torch
import numpy as np
import os


class DataProvider(BasicDataProvider):
    def __init__(self, args, scale=True, scaler=None, device=None):
        super(DataProvider, self).__init__(args, scale, scaler, device)

    def _load_data(self):
        df = pd.read_csv(self.args.data_path, index_col=0)
        return df.iloc[1:, :308].to_numpy()

    def _split_data(self):
        """
        Split the dataset strictly in time order
        """
        # Train-Valid-Test Split
        train_ratio = 1 - self.args.valid_ratio - self.args.test_ratio
        if train_ratio < 0 or train_ratio > 1:
            raise ValueError('Invalid ratio settings. The sum of train, valid and test ratio must be 1')
        train_end = int(train_ratio * self.data.shape[0])
        valid_end = train_end + int(self.args.valid_ratio * self.data.shape[0])
        return self.data[:train_end], self.data[train_end:valid_end], self.data[valid_end:], self.data[:valid_end]

    def _prepare_data(self, data_type):
        """
        :args: seq_len, pred_len
        :return:
        """
        assert data_type in ['train', 'valid', 'test', "train-valid", "all"]
        end_index = len(self.dataset_dict[data_type]) - self.args.seq_len - self.args.pred_len + 1
        x_shape = (end_index, self.args.seq_len, self.dataset_dict[data_type].shape[-1])
        y_shape = (end_index, self.args.pred_len, self.dataset_dict[data_type].shape[-1])
        # Preallocate memory
        x = torch.zeros(x_shape, dtype=torch.float, device=self.device)
        y = torch.zeros(y_shape, dtype=torch.float, device=self.device)
        for i in range(end_index):
            x[i] = torch.tensor(self.dataset_dict[data_type][i:i + self.args.seq_len], device=self.device)
            y[i] = torch.tensor(
                self.dataset_dict[data_type][i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len],
                device=self.device)
        return x, y