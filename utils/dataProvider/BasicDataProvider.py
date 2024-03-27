from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class BasicDataProvider(Dataset):
    def __init__(self, args, scale=True, scaler=None):
        super(BasicDataProvider, self).__init__()
        self.args = args
        self.scale = scale
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.data = self._load_data()  # load data to memory

    def __len__(self):
        return len(self.data)

    def _load_data(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError

    def inverse_transform(self):
        raise NotImplementedError



