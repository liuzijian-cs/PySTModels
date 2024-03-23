from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import numpy as np
import os


class DataPEMS(Dataset):
    def __init__(self, args, data_type, scale=True, scaler=None):
        assert data_type in ['train', 'valid', 'test']
        super(DataPEMS, self).__init__()

        self.args = args
        self.data_type = data_type
        self.scale = scale
        self.scaler = scaler if scaler else StandardScaler()
        self.data = self._data_read_pems()

    def _data_read_pems(self):
        """
        Read dataset
        """
        data_path = os.path.join(self.args.data_path)  # Use os.path.join: Adapt it to the windows or linux file system
        data = np.load(data_path, allow_pickle=True)  # allow_pickle should be true, when reading .npy or .npz files
        data = data['data'][:, :, 0]  # Only the first dimension, traffic flow data
        # Data read successful. For example : PEMS04, data.size = (16992,307)

        # Data split:
        train_ratio = 1 - self.args.valid_ratio - self.args.test_ratio
        if train_ratio > 0 or train_ratio < 1:
            raise ValueError('Invalid ratio settings. The sum of train, valid and test ratio must be 1')

        train_end = int(train_ratio * data.shape[0])
        valid_end = train_end + int(self.args.valid_ratio * data.shape[0])

        if self.data_type == 'train':
            data = data[:train_end]
        elif self.data_type == 'valid':
            data = data[train_end: valid_end]
        else:
            data = data[valid_end:]

        if self.scale:
            self.scaler.fit(data)  # The scaler is trained using data from Data
            data = self.scaler.transform(data)  # Transform (or scale) the data using the scaler

        return data

    def __len__(self):
        return len(self.data) - self.args.seq_len - self.args.pred_len + 1

    def __getitem__(self, index):
        seq_x = self.data[index: index + self.args.seq_len]
        seq_y = self.data[
            index + self.args.seq_len - self.args.label_len, index + self.args.seq_len + self.args.pred_len]
        # TODO
        #         seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        #         seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        #         return seq_x, seq_y, seq_x_mark, seq_y_mark
        return seq_x, seq_y

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
