from sklearn.preprocessing import StandardScaler
from utils.base_function import Color, print_log
from torch.utils.data import Dataset
import numpy as np
import torch
import time


class BasicDataProvider(Dataset):
    def __init__(self, args, scaler=None, device=None):
        super(BasicDataProvider, self).__init__()
        time_start = time.time()
        # 1. variable
        self.data_path = args.data_path
        self.device = device
        self.valid_ratio = args.valid_ratio
        self.test_ratio = args.test_ratio
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.num_workers = args.num_workers
        self.log_file = args.log_file
        # 2. Initialization
        self.scaler = StandardScaler() if scaler is None else scaler
        data = self._load_data()
        t2 = time.time()
        print_log(self.log_file,
                  f"{Color.P}DataProvider[init] ({(t2 - time_start):6.2f}s):{Color.RE} Load dataset {Color.B}{args.data}{Color.RE}, shape: {Color.C}{data.shape}{Color.RE}, max = {Color.C}{data.max()}{Color.RE}, min = {Color.C}{data.min()}{Color.RE}, mean = {Color.C}{data.mean()}{Color.RE}, median = {Color.C}{np.median(data)}{Color.RE}")
        data_train, data_valid, data_test = self._split_data(data)
        self.scaler.fit(data_train)
        self.data_train = self.scaler.transform(data_train)
        self.data_valid = self.scaler.transform(data_valid)
        self.data_test = self.scaler.transform(data_test)
        self.data_dict = {"train": self.data_train, "valid": self.data_valid, "test": self.data_test}
        print_log(self.log_file,
                  f"{Color.P}DataProvider[init] ({(time.time() - t2):6.2f}s):{Color.RE} Scaler: {Color.B}{self.scaler}{Color.RE}, {Color.G}Train:{self.data_train.shape}{Color.RE}, {Color.Y}Valid:{self.data_valid.shape}{Color.RE}, {Color.R}Test:{self.data_test.shape}{Color.RE} ")

    def _load_data(self):
        """
        * EN: Complete the data read logic.
        * CH: 请完成数据读取逻辑。
        data -> self.data
        :param self.data_path: 数据输入路径
        :return data
        """
        raise NotImplementedError

    def _split_data(self, data):
        """
        * EN: Complete the data segmentation and preprocessing logic.
        * CH: 请完成数据分割与预处理逻辑
        :return data_train, data_valid, data_test
        """
        raise NotImplementedError

    def _prepare_data(self, data_type):
        """
        * EN: Complete the logic to get the input and output data.
        * CH: 请完成获取输入和输出数据的逻辑
        :return: X: tensor, Y
        """
        raise NotImplementedError

    def data_loader(self, data_type):
        """
        * EN: Complete the data reading, processing, and returning a dataloader
        * CH: 实现数据读取、处理并返回dataloader`
        :return: train_loader, valid_loader, test_loader
        """
        assert data_type in ["train", "valid", "test"]
        t1 = time.time()
        x, y = self._prepare_data(data_type)
        data = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        name_dict = {
            "train": f"{Color.G}Train data{Color.RE}",
            "valid": f"{Color.Y}Valid data{Color.RE}",
            "test": f"{Color.R}Test data{Color.RE}"
        }
        print_log(self.log_file,
                  f"{Color.P}DataProvider[prep] ({(time.time() - t1):6.2f}s):{Color.RE} {name_dict[data_type]} : x: {Color.C}{x.shape}{Color.RE}, y: {Color.C}{y.shape}{Color.RE}, steps: {Color.C}{len(dataloader)}{Color.RE}, batch_size: {Color.B}{self.batch_size}{Color.RE}, shuffle: {Color.B}{self.shuffle}{Color.RE}, num_workers: {Color.B}{self.num_workers}{Color.RE}")
        return dataloader

    def transform(self, data):
        original_shape = data.shape  # Saves the shape of the original data 保存原始数据的形状
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()  # Ensure the data is on CPU and converted to numpy for the scaler
        data_reshaped = data.reshape(-1, data.shape[-1])
        normalized_data = self.scaler.transform(data_reshaped).reshape(original_shape)
        return normalized_data

    def inverse_transform(self, data):
        original_shape = data.shape
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data_reshaped = data.reshape(-1, data.shape[-1])
        restored_data = self.scaler.inverse_transform(data_reshaped).reshape(original_shape)
        return restored_data
