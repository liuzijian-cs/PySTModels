import torch
import torch.nn as nn
import numpy as np
import time

from utils.model_task.BasicModel import BasicModel


class TimeSeriesForcast(BasicModel):
    def __init__(self, args, model_dict):
        super(TimeSeriesForcast, self).__init__(args, model_dict)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)

        # TODO:multigpu
        # if self.args.use_multi_gpu and self.args.device == 'cuda':
        #     model = nn.DataParallel(model, device_ids=self.args.gpu_ids)

        return model

    def train(self):
        # Read Data:
        train_loader, val_loader, test_loader = self._get_data_loader('train')

        # Train Initialization:
        time_train_start = time.time()
        train_steps = len(train_loader)

        # Optimizer and Criterion: TODO
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args)
        self.model_criterion = torch.nn.MSELoss()

        # AMP - Automatic Mixed Precision : High training speed and reduced memory required by the model
        if self.args.amp:
            amp_scaler = torch.cuda.amp.GradScaler()

        # Train:
        for epoch in range(self.args.epochs):
            train_loss = []  # TODO
            self.model.train()  # Adjust the model to training mode
            for i, (x_batch, y_batch, x_batch_mask, y_batch_mask) in enumerate(train_loader):
                self.model_optimizer.zero_grad()
                # Data: to device
                x_batch = x_batch.float().to(self.args.device)
                y_batch = y_batch.float().to(self.args.device)
                if x_batch_mask is not None:
                    x_batch_mask = x_batch_mask.float().to(self.args.device)
                    y_batch_mask = y_batch_mask.float().to(self.args.device)


