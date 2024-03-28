from utils.base_function import print_log, Color
from utils.dataProviders.BasicDataProvider import BasicDataProvider
import torch
import numpy as np
import os


class DataProvider(BasicDataProvider):
    def __init__(self, args, scale=True, scaler=None):
        super(DataProvider, self).__init__(args, scale, scaler)

    def _load_data(self):
        return np.load(os.path.join(self.args.data_path), allow_pickle=True)['data'][:, :, 0]  # traffic flow

    def _prepare_data(self):
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

    def transform(self, data, data_type="train-valid"):
        super().transform(data, data_type)

    def inverse_transform(self, data, data_type="train-valid"):
        super().inverse_transform(data, data_type)

    def __len__(self):
        return len(self.data) - self.args.seq_len - self.args.pred_len + 1


