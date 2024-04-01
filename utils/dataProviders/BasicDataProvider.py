from sklearn.preprocessing import StandardScaler
from utils.base_function import print_log, Color
from torch.utils.data import Dataset
import numpy as np
import torch
import time


class BasicDataProvider(Dataset):
    def __init__(self, args, scale=True, scaler=None, device=None):
        super(BasicDataProvider, self).__init__()
        # 1. variable initialization
        t1 = time.time()
        self.args = args
        self.device = device
        # 2. load data
        t2 = time.time()
        self.data = self._load_data()
        print_log(self.args,
                  f"{Color.P}DataProvider[init] ({(t2 - t1):6.2f}s):{Color.RE} load dataset {Color.B}{self.args.data}{Color.RE}, shape: {Color.C}{self.data.shape}{Color.RE}, max = {self.data.max()}, min = {self.data.min()}, mean = {self.data.mean()}, median = {np.median(self.data)}")
        # 3. split data
        self.data_train, self.data_valid, self.data_test, self.data_train_valid = self._split_data()
        self.dataset_dict = {
            "train": self.data_train,
            "valid": self.data_valid,
            "test": self.data_test,
            "train-valid": self.data_train_valid,
            "all": self.data  # Default
        }
        t3 = time.time()
        print_log(self.args,
                  f"{Color.P}DataProvider[init] ({(t3 - t2):6.2f}s):{Color.RE} Train:{Color.C}{self.data_train.shape}{Color.RE}, Valid:{Color.C}{self.data_valid.shape}{Color.RE}, Test:{Color.C}{self.data_test.shape}{Color.RE}")
        # 4. scaler initialization
        self.scale = scale
        self.scaler = (scaler if scaler is not None else StandardScaler()).fit(self.data_train_valid)
        self.scaler_train = (scaler if scaler is not None else StandardScaler()).fit(self.data_train)
        self.scaler_valid = (scaler if scaler is not None else StandardScaler()).fit(self.data_valid)
        self.scaler_test = (scaler if scaler is not None else StandardScaler()).fit(self.data_test)
        self.scaler_all = (scaler if scaler is not None else StandardScaler()).fit(self.data)
        self.data_type_dict = {
            "train": self.scaler_train,
            "valid": self.scaler_valid,
            "test": self.scaler_test,
            "train-valid": self.scaler,  # Default
            "all": self.scaler_all
        }
        # 5. scale data
        if scale is True:
            self.data_train = self.transform(self.data_train, "train")
            self.data_valid = self.transform(self.data_valid, "valid")
            self.data_test = self.transform(self.data_test, "test")
            print_log(self.args,
                      f"{Color.P}DataProvider[init] ({(time.time() - t3):6.2f}s):{Color.RE} Data normalization is complete, using {Color.B}{self.scaler}{Color.RE}. ")

    def __len__(self):
        return len(self.data)

    def _load_data(self):
        """
        * EN: Complete the data read logic.
        * CH: 请完成数据读取逻辑。
        data -> self.data
        """
        raise NotImplementedError

    def _split_data(self):
        """
        * EN: Complete the data segmentation and preprocessing logic.
        * CH: 请完成数据分割与预处理逻辑
        self.data -> self.data_train, self.data_valid, self.data_test
        """
        raise NotImplementedError

    def _prepare_data(self, data_type):
        """
        * EN: Complete the logic to get the input and output data.
        * CH: 请完成获取输入和输出数据的逻辑
        :return: X, Y
        """
        raise NotImplementedError

    def train_loader(self):
        t1 = time.time()
        x, y = self._prepare_data("train")
        t2 = time.time()
        print_log(self.args,
                  f"{Color.P}DataProvider[prep] ({(t2 - t1):6.2f}s):{Color.RE} Train data preparation is complete, train_x: {Color.C}{x.shape}{Color.RE}, train_y: {Color.C}{y.shape}{Color.RE}")
        data = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=self.args.batch_size, shuffle=self.args.shuffle, num_workers=self.args.num_workers)
        print_log(self.args,
                  f"{Color.P}DataProvider[prep] ({(time.time() - t2):6.2f}s):{Color.RE} Train dataloader preparation is complete, steps: {Color.C}{len(dataloader)}{Color.RE}, batch_size: {Color.B}{self.args.batch_size}{Color.RE}, shuffle: {Color.B}{self.args.shuffle}{Color.RE}, num_workers: {Color.B}{self.args.num_workers}{Color.RE} ")
        return dataloader

    def valid_loader(self):
        t1 = time.time()
        train_x, train_y = self._prepare_data("valid")
        print_log(self.args,
                  f"{Color.P}DataProvider[prep] ({(time.time() - t1):6.2f}s):{Color.RE} Valid data preparation is complete, valid_x: {Color.C}{len(train_x)}{Color.RE}, valid_y: {Color.C}{len(train_y)}{Color.RE}")
        data = torch.utils.data.TensorDataset(train_x, train_y)
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=self.args.batch_size, shuffle=self.args.shufflem, num_workers=self.args.num_workers)
        print_log(self.args,
                  f"{Color.P}DataProvider[prep] ({(time.time() - t1):6.2f}s):{Color.RE} Valid dataloader preparation is complete, Validloader: size {Color.C}{len(dataloader)}{Color.RE} in {Color.B}{dataloader}{Color}")
        return dataloader

    def test_loader(self):
        t1 = time.time()
        test_x, test_y = self._prepare_data("test")
        print_log(self.args,
                  f"{Color.P}DataProvider[prep] ({(time.time() - t1):6.2f}s):{Color.RE} Test data preparation is complete, test_x: {Color.C}{len(test_x)}{Color.RE}, test_y: {Color.C}{len(test_y)}{Color.RE}")
        data = torch.utils.data.TensorDataset(test_x, test_y)
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=self.args.batch_size, shuffle=self.args.shufflem, num_workers=self.args.num_workers)
        print_log(self.args,
                  f"{Color.P}DataProvider[prep] ({(time.time() - t1):6.2f}s):{Color.RE} Test dataloader preparation is complete, Testloader: size {Color.C}{len(dataloader)}{Color.RE} in {Color.B}{dataloader}{Color}")
        return dataloader

    def transform(self, data, data_type="train-valid"):
        """
        * EN: Normalize the data and keep the dimensions constant.
        If no distribution is given, the train-valid distribution is assumed.
        * CH: 归一化数据，并保证数据维度不变。若不输入数据分布，默认为train-valid分布。
        :param data: input data.
        :param data_type: input data distribution flag 数据分布标志（train, valid, test, train-valid, all）
        :return data: transformed data
        """
        assert data_type in ["train", "valid", "test", "train-valid", "all"]
        original_shape = data.shape  # Saves the shape of the original data 保存原始数据的形状
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()  # Ensure the data is on CPU and converted to numpy for the scaler
        data_reshaped = data.reshape(-1, data.shape[-1])
        normalized_data = self.data_type_dict[data_type].transform(data_reshaped).reshape(original_shape)
        return normalized_data

    def inverse_transform(self, data, data_type="train-valid"):
        """
        * EN: Inverse normalize the data and keep the dimensions constant.
        If no distribution is given, the train-valid distribution is assumed.
        * CH: 对归一化的数据进行逆操作，恢复到原始的数据尺度，并保证数据维度不变。若不输入数据分布，默认为train-valid分布。
        :param data: Normalized data 归一化后的数据
        :param data_type: 数据分布标志（train, valid, test, train-valid, all）
        :return data: 逆变换后的数据
        """
        assert data_type in ["train", "valid", "test", "train-valid", "all"]
        original_shape = data.shape
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data_reshaped = data.reshape(-1, data.shape[-1])
        restored_data = self.data_type_dict[data_type].transform(data_reshaped)
        return restored_data.reshape(original_shape)

    def get_dataset(self, data_type="all"):
        """
        * EN: Get the original dataset.
        * CH: 获取原始数据集。
        """
        assert data_type in ["train", "valid", "test", "train-valid", "all"]
        return self.dataset_dict[data_type]
