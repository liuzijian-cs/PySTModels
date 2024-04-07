import os

import numpy as np
import matplotlib.pyplot as plt
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
            "train": {"loss": [], "mae": [], "rmse": [], "mape": [], "mae_": []},
            "valid": {"loss": [], "mae": [], "rmse": [], "mape": [], "mae_": []},
            "test": {"loss": [], "mae": [], "rmse": [], "mape": [], "mae_": []}
        }

    @staticmethod
    def _mae(y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

    @staticmethod
    def _mape(y_true, y_pred):
        return torch.mean(torch.abs((y_true - y_pred) / y_true))

    @staticmethod
    def _rmse(y_true, y_pred):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

    def _inverse_scale(self, y_pred, y_true) -> np.array:
        return self.DataProvider.inverse_transform(y_pred), self.DataProvider.inverse_transform(y_true)

    def _calculate_evaluation(self, y_pred, y_true):
        mae = self._mae(y_true, y_pred).item()
        rmse = self._rmse(y_true, y_pred).item()
        mape = self._mape(y_true, y_pred).item()
        # inverse scale:
        y_pred = torch.tensor(self.DataProvider.inverse_transform(y_pred)).to(self.device)
        y_true = torch.tensor(self.DataProvider.inverse_transform(y_true)).to(self.device)
        mae_inv = self._mae(y_true, y_pred).item()
        return mae, mae_inv, rmse, mape

    def test(self):
        None

    def valid(self):
        iter_loss, iter_mae, iter_rmse, iter_mape, iter_mae_inverse = [], [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (batch_x, y_true) in enumerate(self.valid_loader):
                batch_x, y_true = batch_x.to(self.device), y_true.to(self.device)
                if self.args.amp and self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        y_pred, attention_weight = self.model(batch_x)  # [B, pred_len, N]
                else:
                    y_pred, attention_weight = self.model(batch_x)  # [B, pred_len, N]

                valid_loss = self.model_criterion(y_pred, y_true).item()
                mae, mae_inverse_scale, rmse, mape = self._calculate_evaluation(y_pred, y_true)
                # TODO
                # y_pred = y_pred.detach().cpu().numpy()
                # print(f"max={y_pred.max():.4f}, min={y_pred.min():.4f}, mean={y_pred.mean():.4f}")
                # y_pred = self.DataProvider.inverse_transform(y_pred)
                # print(f"max={y_pred.max():.4f}, min={y_pred.min():.4f}, mean={y_pred.mean():.4f}")
                # Record:
                iter_loss.append(valid_loss)
                iter_mae.append(mae)
                iter_rmse.append(rmse)
                iter_mape.append(mape)
                iter_mae_inverse.append(mae_inverse_scale)
                #  BATCH for END
            # Record:
            epoch_loss = np.average(iter_loss)
            epoch_mae = np.average(iter_mae)
            epoch_rmse = np.average(iter_rmse)
            epoch_mape = np.average(iter_mape)
            epoch_mae_inverse = np.average(iter_mae_inverse)
        return epoch_loss, epoch_mae, epoch_rmse, epoch_mape, epoch_mae_inverse

    def train(self):
        t1 = time.time()
        print_log(self.log_file, f"{Color.G}>>> Start training.{Color.RE}")
        iter_count = 0
        for epoch in range(self.args.epochs):
            self.model.train()
            iter_loss, iter_mae, iter_rmse, iter_mape, iter_mae_inverse, = [], [], [], [], []
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
                mae, mae_inverse_scale, rmse, mape = self._calculate_evaluation(y_pred, y_true)

                if self.args.log_interval_iter and batch_idx % self.args.log_interval_iter == 0:
                    print_log(self.log_file,
                              f"{Color.P}epoch{epoch:03d}-{batch_idx:03d} {Color.RE}: loss:{loss_item:.6f}, MAE(inverse):{mae_inverse_scale:.6f}({mae:.6f}), rmse:{rmse:.6f}, mape:{mape:.6f}")
                iter_count += 1
                # Save iter record
                iter_loss.append(loss_item)
                iter_mae.append(mae)
                iter_rmse.append(rmse)
                iter_mape.append(mape)
                iter_mae_inverse.append(mae_inverse_scale)
                # BATCH for END

            # Record:
            epoch_loss = np.average(iter_loss)
            epoch_mae = np.average(iter_mae)
            epoch_rmse = np.average(iter_rmse)
            epoch_mape = np.average(iter_mape)
            epoch_mae_inverse = np.average(iter_mae_inverse)
            print(
                f"{Color.G}Train epoch-{epoch:03d}:{Color.RE} Train loss:{Color.C}{epoch_loss:.6f}{Color.RE}, MAE(inverse): {Color.C}{epoch_mae:.6f}({epoch_mae_inverse:.6f}){Color.RE}, RMSE: {Color.C}{epoch_rmse:.6f}{Color.RE}, MAPE:{Color.C}{epoch_mape:.6f}{Color.RE}")

            self.record['train']['loss'].append(loss_item)
            self.record['train']['mae'].append(mae)
            self.record['train']['rmse'].append(rmse)
            self.record['train']['mape'].append(mape)
            self.record['train']['mae_'].append(mae_inverse_scale)
            # VALID:
            valid_loss, valid_mae, valid_rmse, valid_mape, valid_mae_ = self.valid()
            print(
                f"{Color.Y}Valid epoch-{epoch:03d}:{Color.RE} Valid loss:{Color.C}{valid_loss:.6f}{Color.RE}, MAE(inverse): {Color.C}{valid_mae:.6f}({valid_mae_:.6f}){Color.RE}, RMSE: {Color.C}{valid_rmse:.6f}{Color.RE}, MAPE:{Color.C}{valid_mape:.6f}{Color.RE}")

    def draw_predictions(self, data_type="all"):
        assert data_type in ["train", "valid", "test", "all"]
        ground_truth = self.DataProvider.get_data(data_type=data_type)
        ground_truth = self.DataProvider.transform(ground_truth)
        dataloader = self.DataProvider.data_loader(data_type=data_type)
        predictions_matrix = np.zeros((ground_truth.shape[0], ground_truth.shape[1], self.args.pred_len))
        print(predictions_matrix.shape)

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (batch_x, y_true) in enumerate(dataloader):
                batch_x, y_true = batch_x.to(self.device), y_true.to(self.device)
                # Get y_true, attention_weight
                if self.args.amp and self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        y_pred, attention_weight = self.model(batch_x)  # [B, pred_len, N]
                else:
                    y_pred, attention_weight = self.model(batch_x)  # [B, pred_len, N]
                true = y_true.detach().cpu().numpy()
                pred = y_pred.detach().cpu().numpy()
                input = batch_x.detach().cpu().numpy()
                shape = input.shape
                input = self.DataProvider.inverse_transform(input)
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                plt.figure()
                plt.plot(gt, label='GroundTruth', linewidth=2)
                plt.plot(pd, label='Prediction', linewidth=2)
                plt.legend()
                plt.savefig(os.path.join(os.path.join(self.args.pic_save_path, f"batch_{batch_idx}.png")), bbox_inches='tight')


        #         y_pred = y_pred.detach().cpu().numpy()
        #         y_true = np.transpose(y_pred, (0, 2, 1))
        #         # y_pred = self.DataProvider.inverse_transform(y_pred)
        #         start_idx = self.args.batch_size * batch_idx
        #         end_idx = start_idx + y_pred.shape[0]
        #         for t in range(self.args.pred_len):
        #             # predictions_matrix[start_idx:end_idx, :, t] += y_pred[:, t, :].detach().cpu().numpy()
        #             predictions_matrix[start_idx, :, t] = y_pred[:, :, t]
        #
        # for t in range(self.args.pred_len):
        #     predictions_matrix[:, :, t] /= np.count_nonzero(predictions_matrix[:, :, t], axis=0)

        # prediction = np.mean(predictions_matrix, axis=2)
        #
        # # prediction = np.mean(prediction, axis=0)  # 如果有多个值，计算其平均值
        # # ground_truth = np.mean(ground_truth, axis=0)
        #
        # os.makedirs(self.args.pic_save_path, exist_ok=True)
        # for i in range(prediction.shape[1]):  # 假设第二维是不同的变量
        #     plt.figure(figsize=(20, 6))
        #     plt.plot(ground_truth[:, i], label='Ground Truth', color='blue')
        #     plt.plot(prediction[:, i], label='Prediction', color='red')
        #     plt.title(f'Variable {i} Prediction')
        #     plt.legend()
        #     save_path = os.path.join(self.args.pic_save_path, f"variable_{i}.png")
        #     plt.savefig(save_path)
        #     plt.close()  # 关闭图像，避免内存泄漏

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
