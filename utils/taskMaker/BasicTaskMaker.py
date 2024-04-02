from utils.base_function import print_log, Color
import os
import time
import torch
import numpy as np


class BasicTask:
    def __init__(self, args, model_dict, data_dict):
        t1 = time.time()
        self.args = args
        self.model_dict = model_dict
        self.data_dict = data_dict
        self.amp_scaler = None
        self.device = self._prepare_device_seed()
        t2 = time.time()
        print_log(self.args,
                  f"{Color.P}TaskMaker[init]    ({(t2 - t1):6.2f}s):{Color.RE} Deivce and seed initialization is complete, using device: {Color.B}{self.device}{Color.RE}, using seed: {Color.B}{self.args.seed}{Color.RE} ")

        self.model = self._prepare_model().to(self.device)
        t3 = time.time()
        print_log(self.args,
                  f"{Color.P}TaskMaker[init]    ({(t3 - t2):6.2f}s):{Color.RE} Model {Color.B}{self.args.model}{Color.RE} initialization is complete, Number of parameters: {Color.C}{sum([p.nelement() for p in self.model.parameters()])}{Color.RE}")
        self.DataProvider = self.data_dict[self.args.data].DataProvider(
            self.args, self.args.scale, self.args.scaler,self.device)
        t4 = time.time()
        print_log(self.args,
                  f"{Color.P}TaskMaker[init]    ({(t4 - t3):6.2f}s):{Color.RE} Created DataProvider: {Color.B}{self.data_dict[self.args.data]}")
        self.train_loader = self.DataProvider.data_loader("train")
        self.valid_loader = self.DataProvider.data_loader("valid")
        self.test_loader = self.DataProvider.data_loader("test")
        print_log(self.args,
                  f"{Color.P}TaskMaker[init]    ({(time.time() - t4):6.2f}s):{Color.RE} Created dataloader! {Color.G}Train: {len(self.train_loader)}{Color.RE}, {Color.Y}Valid: {len(self.valid_loader)}{Color.RE}, {Color.R}Test: {len(self.test_loader)}{Color.RE}. {Color.P}TaskMaker initialization is complete, time cost: ({(time.time() - t1):6.2f}s):{Color.RE}.")

        # Needs to be implemented in subclass:
        self.model_optimizer = None
        self.model_criterion = None



    def _prepare_device_seed(self):
        """
        * EN: Initialize the device and the random seed
        * CH: 初始化设备和随机种子
        :return device
        """
        # Device initialization:
        if self.args.device == 'cuda':
            os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu_id if not self.args.use_multi_gpu else self.args.gpu_ids
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f'{Color.R}cuda is not available! (args.device == {Color.B}{self.args.deivce}{Color.R}){Color.RE}')
            device = torch.device(f'cuda:{self.args.gpu_id}')
        else:
            device = torch.device('cpu')
        # Seed initialization:
        if self.args.seed is not None:
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            if self.args.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.init()
                torch.cuda.manual_seed_all(self.args.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        # AMP - Automatic Mixed Precision : High training speed and reduced memory required by the model
        if self.args.amp and device == 'cuda':
            self.amp_scaler = torch.cuda.amp.GradScaler()
        return device

    def _prepare_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.use_multi_gpu and self.args.device == 'cuda':
            model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
        return model

    def train(self):
        raise NotImplementedError

    # def valid(self):
    #     raise NotImplementedError
    #
    # def test(self):
    #     raise NotImplementedError


