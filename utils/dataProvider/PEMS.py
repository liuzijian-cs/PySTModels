from utils.base_function import print_log
from utils.dataProvider.BasicDataProvider import BasicDataProvider
import numpy as np
import os


class PEMS(BasicDataProvider):
    def __init__(self, args, scale=True, scaler=None):
        super(PEMS, self).__init__(args, scale, scaler)
        None

    def _load_data(self):
        self.data = np.load(os.path.join(self.args.data_path))['data'][:, :, 0]  # traffic flow
        print_log(self.args,
                  f"DataProvider       : load data PEMS from {self.args.data}, shape: {self.data.shape}, max = {self.data.max()}, min = {self.data.min()}, mean = {self.data.mean()}, median = {np.median(self.data)}")
