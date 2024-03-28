


class BasicModel(object):
    def __init__(self, args, model_dict):
        self.args = args
        self.device = args.device
        self.model_dict = model_dict
        self.model = self._build_model().to(self.device)\

        self.model_optimizer = None
        self.model_criterion = None

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self, data_type):
        data, _ = data_loader(self.args, data_type)
        return data

    def _get_data_loader(self, data_type):
        _, dataloader = data_loader(self.args, data_type)
        return dataloader

    def _get_data_loaders(self):
        _, train_loader = data_loader(self.args, data_type='train')
        _, valid_loader = data_loader(self.args, data_type='valid')
        _, test_loader = data_loader(self.args, data_type='test')
        return train_loader, valid_loader, test_loader

    def train(self):
        raise NotImplementedError

    def valid(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
