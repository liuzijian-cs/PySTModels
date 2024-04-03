from utils.taskMaker.BasicTaskMaker import BasicTask
from utils.base_function import print_log, Color
import time
import torch
import torch.nn as nn


class Task(BasicTask):
    def __init__(self, args, model_dict, data_dict):
        super().__init__(args, model_dict, data_dict)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model_criterion = torch.nn.MSELoss()
        # Records:
        # LOSS
        self.train_loss_epoch = []
        self.train_loss_iter = []
        self.valid_loss_epoch = []
        self.valid_loss_iter = []
        self.test_loss_epoch = []
        self.test_loss_iter = []
        # MAE
        self.train_mae_epoch = []
        self.train_mae_iter = []
        self.valid_mae_epoch = []
        self.valid_mae_iter = []
        self.test_mae_epoch = []
        self.test_mae_iter = []
        #
        self.train_mae_epoch = []
        self.train_mae_iter = []
        self.valid_mae_epoch = []
        self.valid_mae_iter = []
        self.test_mae_epoch = []
        self.test_mae_iter = []

    @staticmethod
    def _mae(y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

    @staticmethod
    def _mape(y_true, y_pred):
        return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def _rmse(y_true, y_pred):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

    def train(self):
        train_loader = self.DataProvider.data_loader("train")
        valid_loader = self.DataProvider.data_loader("valid")
        test_loader = self.DataProvider.data_loader("test")
        # print_log(self.args,
        #           f"{Color.P}TaskMaker[init]    ({(time.time() - t4):6.2f}s):{Color.RE} Created dataloader! {Color.G}Train: {len(self.train_loader)}{Color.RE}, {Color.Y}Valid: {len(self.valid_loader)}{Color.RE}, {Color.R}Test: {len(self.test_loader)}{Color.RE}. {Color.P}TaskMaker initialization is complete, time cost: ({(time.time() - t1):6.2f}s):{Color.RE}.")

        t1 = time.time()
        print_log(self.args,
                  f"{Color.G}>>> ({(time.time() - t1):6.2f}s) Start training.{Color.RE}")
        iter_count = 0
        for epoch in range(self.args.epochs):
            self.model.train()
            time_epoch = time.time()
            for batch_idx, (batch_x, y_true) in enumerate(train_loader):
                print(
                    f'After to(device): batch_x stats - max: {batch_x.max()}, min: {batch_x.min()}, mean: {batch_x.mean()}')
                self.model_optimizer.zero_grad()
                # Data to device
                batch_x, y_true = batch_x.to(self.device), y_true.to(self.device)
                if self.args.amp and self.args.device == "cuda":
                    with torch.cuda.amp.autocast():
                        y_pred, attention_weight = self.model(batch_x)  # [B, pred_len, N]
                        loss = self.model_criterion(y_pred,y_true)
                else:
                    y_pred, attention_weight = self.model(batch_x)
                    loss = self.model_criterion(y_pred, y_true)

                if self.args.data_inverse_scale:
                    y_true_inverse = torch.tensor(self.DataProvider.inverse_transform(y_true)).to(self.device)
                    y_pred_inverse = torch.tensor(self.DataProvider.inverse_transform(y_pred)).to(self.device)
                    mae = self._mae(y_true_inverse,y_pred_inverse)
                    mape = self._mape(y_true_inverse,y_pred_inverse)
                    rmse = self._rmse(y_true_inverse,y_pred_inverse)
                else:
                    mae = self._mae(y_true,y_pred)
                    mape = self._mape(y_true,y_pred)
                    rmse = self._rmse(y_true,y_pred)

                loss_item = loss.item()
                print(f"epoch{epoch}-{batch_idx}: loss:{loss_item:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}")

                # Backward and optimize
                if self.args.amp and self.args.device == "cuda":
                    self.amp_scaler.scale(loss).backward()
                    self.amp_scaler.step(self.model_optimizer)
                    self.amp_scaler.update()
                else:
                    loss.backward()
                    self.model_optimizer.step()
