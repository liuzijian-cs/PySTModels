import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.taskMaker.BasicTaskMaker import BasicTask
from utils.base_function import Color, print_log
from utils.draw_function import draw_record
import time
import torch
from tqdm import tqdm


# tqdm:
bar_format = '{l_bar}{bar:30}{r_bar}'


class Task(BasicTask):
    def __init__(self, args, model_dict, data_dict):
        super().__init__(args, model_dict, data_dict)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model_criterion = torch.nn.MSELoss()
        self.record = {
            "train": {"loss": [], "mae": [], "rmse": [], "mape": [], "mae_": []},
            "valid": {"loss": [], "mae": [], "rmse": [], "mape": [], "mae_": []},
        }
        self.time_epoch_start = None
        self.epoch = None  # use in valid, pls don't change this

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
            with tqdm(total=len(self.test_loader),
                      ncols=200,
                      colour='RED',
                      desc=f"{Color.R}Valid epoch-{self.epoch:03d} [{(time.time() - self.time_epoch_start) / 3600:.2f}<{(((time.time() - self.time_epoch_start) / 3600) / (self.epoch + 1)) * self.args.epochs:.2f} hours]:{Color.RE}",
                      bar_format="{desc}{bar:15}[{percentage:3.0f}% {elapsed}<{remaining}] {postfix}",
                      ) as pbar:
                for batch_idx, (batch_x, y_true) in enumerate(self.test_loader):
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
                    pbar.set_postfix_str(
                        f"{Color.R}iter loss: {Color.C}{valid_loss:.4f}{Color.R}, MAE(inverse): {Color.C}{mae:.4f}({mae_inverse_scale:.4f}){Color.R}, RMSE: {Color.C}{rmse:.4f}{Color.R}, MAPE :{Color.C}{mape:.4f}%{Color.RE}",
                        refresh=True)
                    pbar.update(1)
                    #  BATCH for END
                # Record:
                epoch_loss = np.average(iter_loss)
                epoch_mae = np.average(iter_mae)
                epoch_rmse = np.average(iter_rmse)
                epoch_mape = np.average(iter_mape)
                epoch_mae_inverse = np.average(iter_mae_inverse)
                pbar.set_postfix_str(
                    f"{Color.R}Epoch[Test ] : loss: {Color.C}{epoch_loss:.6f}{Color.R}, MAE(inverse): {Color.C}{epoch_mae:.6f}({epoch_mae_inverse:.6f}){Color.R}, RMSE: {Color.C}{epoch_rmse:.6f}{Color.R}, MAPE :{Color.C}{epoch_mape:.6f}%{Color.RE}",
                    refresh=True)
        return epoch_loss, epoch_mae, epoch_rmse, epoch_mape, epoch_mae_inverse

    def valid(self):
        iter_loss, iter_mae, iter_rmse, iter_mape, iter_mae_inverse = [], [], [], [], []
        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(self.valid_loader),
                      ncols=200,
                      colour='YELLOW',
                      desc=f"{Color.Y}Valid epoch-{self.epoch:03d} [{(time.time() - self.time_epoch_start) / 3600:.2f}<{(((time.time() - self.time_epoch_start) / 3600) / (self.epoch + 1)) * self.args.epochs:.2f} hours]:{Color.RE}",
                      bar_format="{desc}{bar:15}[{percentage:3.0f}% {elapsed}<{remaining}] {postfix}",
                      ) as pbar:
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
                    pbar.set_postfix_str(
                        f"{Color.Y}iter loss: {Color.C}{valid_loss:.4f}{Color.Y}, MAE(inverse): {Color.C}{mae:.4f}({mae_inverse_scale:.4f}){Color.Y}, RMSE: {Color.C}{rmse:.4f}{Color.Y}, MAPE :{Color.C}{mape:.4f}%{Color.RE}",
                        refresh=True)
                    pbar.update(1)
                    #  BATCH for END
                # Record:
                epoch_loss = np.average(iter_loss)
                epoch_mae = np.average(iter_mae)
                epoch_rmse = np.average(iter_rmse)
                epoch_mape = np.average(iter_mape)
                epoch_mae_inverse = np.average(iter_mae_inverse)
                pbar.set_postfix_str(
                    f"{Color.Y}Epoch[Valid] : loss: {Color.C}{epoch_loss:.6f}{Color.Y}, MAE(inverse): {Color.C}{epoch_mae:.6f}({epoch_mae_inverse:.6f}){Color.Y}, RMSE: {Color.C}{epoch_rmse:.6f}{Color.Y}, MAPE :{Color.C}{epoch_mape:.6f}%{Color.RE}",
                    refresh=True)
        return epoch_loss, epoch_mae, epoch_rmse, epoch_mape, epoch_mae_inverse

    def train(self):
        t1 = time.time()
        print_log(self.log_file, f"{Color.G}>>> Start training.{Color.RE}")
        time.sleep(0.1)  # For beautiful console
        epoch_list = []

        self.time_epoch_start = time.time()
        for self.epoch in range(self.args.epochs):
            self.model.train()
            iter_loss, iter_mae, iter_rmse, iter_mape, iter_mae_inverse, = [], [], [], [], []

            with tqdm(total=len(self.train_loader),
                      ncols=200,
                      colour='GREEN',
                      desc=f"{Color.G}Train epoch-{self.epoch:03d} [{(time.time() - self.time_epoch_start) / 3600:.2f}<{(((time.time() - self.time_epoch_start) / 3600) / (self.epoch + 1)) * self.args.epochs:.2f} hours]:{Color.RE}",
                      bar_format="{desc}{bar:15}[{percentage:3.0f}% {elapsed}<{remaining}] {postfix}",
                      ) as pbar:
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
                                  f"{Color.P}epoch{self.epoch:03d}-{batch_idx:03d} {Color.RE}: loss:{loss_item:.6f}, MAE(inverse):{mae:.6f}({mae_inverse_scale:.6f}), rmse:{rmse:.6f}, mape:{mape:.6f}")
                    # Save iter record
                    iter_loss.append(loss_item)
                    iter_mae.append(mae)
                    iter_rmse.append(rmse)
                    iter_mape.append(mape)
                    iter_mae_inverse.append(mae_inverse_scale)
                    pbar.set_postfix_str(
                        f"{Color.G}iter loss: {Color.C}{loss_item:.4f}{Color.G}, MAE(inverse): {Color.C}{mae:.4f}({mae_inverse_scale:.4f}){Color.G}, RMSE: {Color.C}{rmse:.4f}{Color.G}, MAPE :{Color.C}{mape:.4f}%{Color.RE}",
                        refresh=True)
                    # pbar.set_postfix(f"{Color.C}{mae:.6f}{Color.RE}", refresh=True)
                    # pbar.set_postfix("sb")
                    pbar.update(1)
                    # BATCH for END
                # with TQDM END

                # Record:
                epoch_list.append(self.epoch + 1)
                epoch_loss = np.average(iter_loss)
                epoch_mae = np.average(iter_mae)
                epoch_rmse = np.average(iter_rmse)
                epoch_mape = np.average(iter_mape)
                epoch_mae_inverse = np.average(iter_mae_inverse)
                pbar.set_postfix_str(
                    f"{Color.G}Epoch[Train] : loss: {Color.C}{epoch_loss:.6f}{Color.G}, MAE(inverse): {Color.C}{epoch_mae:.6f}({epoch_mae_inverse:.6f}){Color.G}, RMSE: {Color.C}{epoch_rmse:.6f}{Color.G}, MAPE :{Color.C}{epoch_mape:.6f}%{Color.RE}",
                    refresh=True)
                print_log(self.log_file,
                          f"{Color.G}Train epoch-{self.epoch:03d}:{Color.RE} Train loss:{Color.C}{epoch_loss:.6f}{Color.RE}, MAE(inverse): {Color.C}{epoch_mae:.6f}({epoch_mae_inverse:.6f}){Color.RE}, RMSE: {Color.C}{epoch_rmse:.6f}{Color.RE}, MAPE:{Color.C}{epoch_mape:.6f}{Color.RE}",False)
            self.record['train']['loss'].append(loss_item)
            self.record['train']['mae'].append(mae)
            self.record['train']['rmse'].append(rmse)
            self.record['train']['mape'].append(mape)
            self.record['train']['mae_'].append(mae_inverse_scale)
            # VALID:
            valid_loss, valid_mae, valid_rmse, valid_mape, valid_mae_ = self.valid()
            print_log(self.log_file,
                      f"{Color.Y}Valid epoch-{self.epoch:03d}:{Color.RE} Valid loss:{Color.C}{valid_loss:.6f}{Color.RE}, MAE(inverse): {Color.C}{valid_mae:.6f}({valid_mae_:.6f}){Color.RE}, RMSE: {Color.C}{valid_rmse:.6f}{Color.RE}, MAPE:{Color.C}{valid_mape:.6f}{Color.RE}",False)
            self.record['valid']['loss'].append(valid_loss)
            self.record['valid']['mae'].append(valid_mae)
            self.record['valid']['rmse'].append(valid_rmse)
            self.record['valid']['mape'].append(valid_mape)
            self.record['valid']['mae_'].append(valid_mae_)

            torch.save(self.model.state_dict(), os.path.join(self.args.model_save_path, f"epoch-{self.epoch:03d}.pth"))
            # === END FOR: EPOCH ===

        # Test:
        test_loss, test_mae, test_rmse, test_mape, test_mae_ = self.test()
        print_log(self.log_file,
                  f"{Color.R}Test :{Color.RE} Test loss:{Color.C}{test_loss:.6f}{Color.RE}, MAE(inverse): {Color.C}{test_mae:.6f}({test_mae_:.6f}){Color.RE}, RMSE: {Color.C}{test_rmse:.6f}{Color.RE}, MAPE:{Color.C}{test_mape:.6f}{Color.RE}")
        # Save Data:
        metric = pd.DataFrame({
            'epoch': epoch_list,
            'train_loss': self.record['train']['loss'],
            'valid_loss': self.record['valid']['loss'],
            'train_mae': self.record['train']['mae'],
            'valid_mae': self.record['valid']['mae'],
            'train_mae_inverse': self.record['train']['mae_'],
            'valid_mae_inverse': self.record['valid']['mae_'],
            'train_rmse': self.record['train']['rmse'],
            'valid_rmse': self.record['valid']['rmse'],
            'train_mape': self.record['train']['mape'],
            'valid_mape': self.record['valid']['mape'],
        })
        metric.to_csv(os.path.join(self.args.model_save_path, "logs", f"record.csv"))
        # Draw:
        draw_record(self.args.pic_save_path, self.args.epochs, self.record)
        # === END FUNCTION: TRAIN ===

    def draw_predictions(self, data_type="all", draw_batch_limit=np.inf):
        t1 = time.time()
        assert data_type in ["train", "valid", "test", "all"]
        dataloader = self.DataProvider.data_loader(data_type=data_type)
        predictions = []
        ground_truths = []

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
                y_pred = y_pred.detach().cpu().numpy().transpose(0, 2, 1)  # [B, P, N] -> [B, N, P]
                predictions.append(y_pred)
                ground_truths.append(y_true.detach().cpu().numpy().transpose(0, 2, 1))
                if batch_idx > draw_batch_limit:
                    break
            # Concatenate along the first axis to get [total_batches * B, N, P]
            predictions = np.concatenate(predictions, axis=0)
            ground_truths = np.concatenate(ground_truths, axis=0)
            # Draw:
            t2 = time.time()
            bar_format = "{desc} {bar:20} {n_fmt}/{total_fmt}[{elapsed}<{remaining} {rate_fmt}]"
            for i in tqdm(range(predictions.shape[1]),
                          colour='BLUE',
                          desc=f"{Color.P}TaskMaker[Drawer]  ({(time.time() - t1):6.2f}s):{Color.RE}{Color.B} Prediction finished, Drawing: {Color.RE}",
                          bar_format=bar_format):  # 假设第二维是不同的变量
                plt.figure(figsize=(96, 24))
                plt.plot(predictions[:, i, 0], label='Ground Truth', color='blue', alpha=0.7, linewidth=3)
                plt.plot(ground_truths[:, i, 0], label='Prediction', color='red', alpha=0.7, linewidth=3)
                plt.title(f'Variable {i} Prediction')
                plt.legend()
                save_path = os.path.join(self.args.pic_save_path, f"variable_{i}.png")
                plt.savefig(save_path)
                plt.close()  # 关闭图像，避免内存泄漏
            print(
                f"{Color.P}TaskMaker[Drawer]  ({(time.time() - t2):6.2f}s):{Color.RE} Drawing finished{Color.RE}")

            # y_pred = self.DataProvider.inverse_transform(y_pred).transpose(0, 2, 1)  # [B, P, N] -> [B, N, P]
        #         y_pred = y_pred.detach().cpu().numpy().transpose(0, 2, 1)
        #         start_idx = self.args.batch_size * batch_idx + self.args.seq_len
        #         for b in range(y_pred.shape[0]):
        #             for p in range(y_pred.shape[2]):
        #                 prediction[start_idx + b + p, :, p] = y_pred[b, :, p]
        # print(f"min:{prediction.min():.4f}, max:{prediction.max():.4f}, mean:{prediction.mean():.4f}")
        # non_zero_count = np.count_nonzero(prediction, axis=2)
        # non_zero_count = np.where(non_zero_count == 0, 1, non_zero_count)
        # prediction = prediction.sum(axis=2) / non_zero_count
        # print(prediction.shape)
        # print(f"min:{prediction.min():.4f}, max:{prediction.max():.4f}, mean:{prediction.mean():.4f}")
        # for i in range(prediction.shape[1]):  # 假设第二维是不同的变量
        #     plt.figure(figsize=(20, 6))
        #     plt.plot(ground_truth[:, i], label='Ground Truth', color='blue')
        #     plt.plot(prediction[:, i], label='Prediction', color='red')
        #     plt.title(f'Variable {i} Prediction')
        #     plt.legend()
        #     save_path = os.path.join(self.args.pic_save_path, f"variable_{i}.png")
        #     plt.savefig(save_path)
        #     plt.close()  # 关闭图像，避免内存泄漏
