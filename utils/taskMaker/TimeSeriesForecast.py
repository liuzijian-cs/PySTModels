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
        iter_loss, iter_mae, iter_rmse, iter_mape, iter_mae_inverse = [], [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (batch_x, y_true) in enumerate(self.test_loader):
                batch_x, y_true = batch_x.to(self.device), y_true.to(self.device)
                if self.args.amp and self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        y_pred, attention_weight = self.model(batch_x)  # [B, pred_len, N]
                else:
                    y_pred, attention_weight = self.model(batch_x)  # [B, pred_len, N]
                valid_loss = self.model_criterion(y_pred, y_true).item()
                mae, mae_inverse_scale, rmse, mape = self._calculate_evaluation(y_pred, y_true)
                iter_loss.append(valid_loss)
                iter_mae.append(mae)
                iter_rmse.append(rmse)
                iter_mape.append(mape)
                iter_mae_inverse.append(mae_inverse_scale)
            # Record:
            epoch_loss = np.average(iter_loss)
            epoch_mae = np.average(iter_mae)
            epoch_rmse = np.average(iter_rmse)
            epoch_mape = np.average(iter_mape)
            epoch_mae_inverse = np.average(iter_mae_inverse)
        return epoch_loss, epoch_mae, epoch_rmse, epoch_mape, epoch_mae_inverse



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
                              f"{Color.P}epoch{epoch:03d}-{batch_idx:03d} {Color.RE}: loss:{loss_item:.6f}, MAE(inverse):{mae:.6f}({mae_inverse_scale:.6f}), rmse:{rmse:.6f}, mape:{mape:.6f}")
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
            print_log(self.log_file,
                      f"{Color.G}Train epoch-{epoch:03d}:{Color.RE} Train loss:{Color.C}{epoch_loss:.6f}{Color.RE}, MAE(inverse): {Color.C}{epoch_mae:.6f}({epoch_mae_inverse:.6f}){Color.RE}, RMSE: {Color.C}{epoch_rmse:.6f}{Color.RE}, MAPE:{Color.C}{epoch_mape:.6f}{Color.RE}")

            self.record['train']['loss'].append(loss_item)
            self.record['train']['mae'].append(mae)
            self.record['train']['rmse'].append(rmse)
            self.record['train']['mape'].append(mape)
            self.record['train']['mae_'].append(mae_inverse_scale)
            # VALID:
            valid_loss, valid_mae, valid_rmse, valid_mape, valid_mae_ = self.valid()
            print_log(self.log_file,
                      f"{Color.Y}Valid epoch-{epoch:03d}:{Color.RE} Valid loss:{Color.C}{valid_loss:.6f}{Color.RE}, MAE(inverse): {Color.C}{valid_mae:.6f}({valid_mae_:.6f}){Color.RE}, RMSE: {Color.C}{valid_rmse:.6f}{Color.RE}, MAPE:{Color.C}{valid_mape:.6f}{Color.RE}")
        # test:
        test_loss, test_mae, test_rmse, test_mape, test_mae_ = self.test()
        print_log(self.log_file,
                  f"{Color.R}Test :{Color.RE} Test loss:{Color.C}{test_loss:.6f}{Color.RE}, MAE(inverse): {Color.C}{test_mae:.6f}({test_mae_:.6f}){Color.RE}, RMSE: {Color.C}{test_rmse:.6f}{Color.RE}, MAPE:{Color.C}{test_mape:.6f}{Color.RE}")

        torch.save(self.model.state_dict(), os.path.join(self.args.model_save_path, f"epoch-{epoch:03d}.pth"))  # 模型保存

    def draw_predictions(self, data_type="all"):
        assert data_type in ["train", "valid", "test", "all"]
        # Get Ground Truth:
        ground_truth = self.DataProvider.get_data(data_type=data_type)
        # ground_truth = self.DataProvider.inverse_transform(ground_truth)
        # Get Prediction:
        dataloader = self.DataProvider.data_loader(data_type=data_type)
        prediction = np.zeros((ground_truth.shape[0] + self.args.pred_len, ground_truth.shape[1], self.args.pred_len))
        # prediction [T + pred_len(P), N, pred_len(P)]
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (batch_x, y_true) in enumerate(dataloader):
                batch_x = batch_x.to(self.device)
                # Get y_true, attention_weight
                if self.args.amp and self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        y_pred, attention_weight = self.model(batch_x)  # [B, pred_len, N]
                else:
                    y_pred, attention_weight = self.model(batch_x)  # [B, pred_len, N]
                # y_pred = self.DataProvider.inverse_transform(y_pred).transpose(0, 2, 1)  # [B, P, N] -> [B, N, P]
                y_pred = y_pred.detach().cpu().numpy().transpose(0,2,1)
                start_idx = self.args.batch_size * batch_idx
                for b in range(y_pred.shape[0]):
                    for p in range(y_pred.shape[2]):
                        prediction[start_idx + b + p, :, p] = y_pred[b, :, p]
        print(f"min:{prediction.min():.4f}, max:{prediction.max():.4f}, mean:{prediction.mean():.4f}")
        non_zero_count = np.count_nonzero(prediction,axis=2)
        non_zero_count = np.where(non_zero_count == 0, 1, non_zero_count)
        prediction = prediction.sum(axis=2) / non_zero_count
        print(prediction.shape)
        print(f"min:{prediction.min():.4f}, max:{prediction.max():.4f}, mean:{prediction.mean():.4f}")
        for i in range(prediction.shape[1]):  # 假设第二维是不同的变量
            plt.figure(figsize=(20, 6))
            plt.plot(ground_truth[:, i], label='Ground Truth', color='blue')
            plt.plot(prediction[:, i], label='Prediction', color='red')
            plt.title(f'Variable {i} Prediction')
            plt.legend()
            save_path = os.path.join(self.args.pic_save_path, f"variable_{i}.png")
            plt.savefig(save_path)
            plt.close()  # 关闭图像，避免内存泄漏
