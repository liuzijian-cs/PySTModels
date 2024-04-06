import numpy as np

from utils.taskMaker.BasicTaskMaker import BasicTask
from utils.base_function import Color, print_log
import time
import torch
import torch.nn as nn


class Task(BasicTask):
    def __init__(self, args, model_dict, data_dict):
        super().__init__(args, model_dict, data_dict)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model_criterion = torch.nn.MSELoss()
        self.record = {
            "train": {"loss": [], "mae": [], "rmse": [], "mape": []},
            "valid": {"loss": [], "mae": [], "rmse": [], "mape": []},
            "test": {"loss": [], "mae": [], "rmse": [], "mape": []}
        }

    @staticmethod
    def _mae(y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

    @staticmethod
    def _mape(y_true, y_pred):
        return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def _rmse(y_true, y_pred):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

    def _inverse_scale(self, y_pred, y_true) -> np.array:
        return self.DataProvider.inverse_transform(y_pred), self.DataProvider.inverse_transform(y_true)

    # def valid(self):
    #

    def train(self):
        t1 = time.time()
        print_log(self.log_file, f"{Color.G}>>> Start training.{Color.RE}")
        iter_count = 0
        for epoch in range(self.args.epochs):
            self.model.train()
            iter_loss, iter_mae, iter_rmse, iter_mape = [], [], [], []
            for batch_idx, (batch_x, y_true) in enumerate(self.train_loader):
                self.model_optimizer.zero_grad()
                batch_x, y_true = batch_x.to(self.device), y_true.to(self.device)

                if self.args.amp and self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        y_pred, attention_weight = self.model(batch_x)  # [B, pred_len, N]
                        loss = self.model_criterion(y_pred, y_true)
                        self.amp_scaler.scale(loss).backward()
                        self.amp_scaler.scaler.step(self.model_optimizer)
                        self.amp_scaler.update()
                else:
                    y_pred, attention_weight = self.model(batch_x)  # [B, pred_len, N]
                    loss = self.model_criterion(y_pred, y_true)
                    loss.backward()
                    self.model_optimizer.step()

                loss_item = loss.item()
                mae = self._mae(y_pred, y_true)
                rmse = self._rmse(y_pred, y_true)
                mape = self._mape(y_pred, y_true)

                if self.args.data_inverse_scale:
                    y_pred = self.DataProvider.inverse_transform(y_pred)
                    y_true = self.DataProvider.inverse_transform(y_true)
                    mae_inverse_scale = self._mae(y_pred, y_true)
                    rmse_inverse_scale = self._rmse(y_pred, y_true)
                    mape_inverse_scale = self._mape(y_pred, y_true)

                if self.args.log_interval_iter and batch_idx % self.args.log_interval_iter == 0:
                    print_log(self.log_file,
                              f"{Color.P}epoch{epoch:03d}-{batch_idx:03d} {Color.RE}: loss:{loss_item:.4f}, mae:{mae_inverse_scale:.4f}({mae:.4f}), rmse:{rmse_inverse_scale:.4f}({rmse:.4f}), mape:{mape_inverse_scale:.4f}({mape:.4f})")
                iter_count += 1
                # Save iter record
                iter_loss.append(loss_item)
                iter_mae.append(mae)
                iter_rmse.append(rmse)
                iter_mape.append(mape)
                # BATCH for END

            # Record:
            epoch_loss = np.average(iter_loss)
            epoch_mae = np.average(iter_mae)
            epoch_rmse = np.average(iter_rmse)
            epoch_mape = np.average(iter_mape)
            epoch_mae_inverse = self.DataProvider.inverse_transform(epoch_mae)  # ? 不知能不能用
            epoch_rmse_inverse = self.DataProvider.inverse_transform(epoch_rmse)
            epoch_mape_inverse = self.DataProvider.inverse_transform(epoch_mape)
            print(f"{epoch_loss},{epoch_mae},{epoch_rmse},{epoch_mape},{epoch_mae_inverse},{epoch_rmse_inverse},{epoch_mape_inverse}")

            self.record['train']['loss'].append(loss_item)
            self.record['train']['mae'].append(mae)
            self.record['train']['rmse'].append(rmse)
            self.record['train']['mape'].append(mape)
            self.record['train']['mae_'].append(mae_inverse_scale)
            self.record['train']['rmse_'].append(rmse_inverse_scale)
            self.record['train']['mape_'].append(mape_inverse_scale)
            # VALID:

#
#
#
#
#
#         # Train:
#         for epoch in range(self.args.epochs):
#             train_loss = []  # TODO
#             self.model.train()  # Adjust the model to training mode
#             for i, (x_batch, y_batch, x_batch_mask, y_batch_mask) in enumerate(train_loader):
#                 self.model_optimizer.zero_grad()
#                 # Data: to device
#                 x_batch = x_batch.float().to(self.args.device)
#                 y_batch = y_batch.float().to(self.args.device)
#                 if x_batch_mask is not None:
#                     x_batch_mask = x_batch_mask.float().to(self.args.device)
#                     y_batch_mask = y_batch_mask.float().to(self.args.device)
#
#
# # if self.args.features == "M":  # Multivariate predicts multivariate.
#     #     None
#     # elif self.args.features == "MS":
#     #     return None  # TODO
#     # elif self.args.features == "S":
#     #     return None  # TODO
#     # else:
#     #     raise RuntimeError(
#     #         f'{Color.R}PEMS.py! : RuntimeError, There is no matching features value, make sure the --features parameter is in {Color.B}[M,MS,S]{Color.R}{Color.RE}')
