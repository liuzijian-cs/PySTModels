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
        t1 = time.time()
        print_log(self.args,
                  f"{Color.G}>>> ({(time.time() - t1):6.2f}s) Start training.{Color.RE}")
        iter_count = 0
        for epoch in range(self.args.epochs):
            self.model.train()
            for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                self.model_optimizer.zero_grad()
                # Data to device
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                if self.args.amp and self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        y_pred, attention_weight = self.model(batch_x)
                else:
                    y_pred, attention_weight = self.model(batch_x)
                print(f"test: y_pred:{y_pred.shape}")
                y_pred_transformed = self.DataProvider.inverse_transform(y_pred)
                y_true_transformed = self.DataProvider.inverse_transform(batch_y)
                print(f"test: y_pred_transformed:{y_pred_transformed.shape}")
                print(f"test: y_true_transformed:{y_true_transformed.shape}")




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
