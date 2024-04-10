import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, args, input_dim=100):
        super(Model, self).__init__()
        self.device = args.device
        self.input_dim = input_dim
        self.hidden_dim = args.d_model
        self.num_layers = args.model_layers
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                            batch_first=True)
        """
        layer - nn.LSTM:
        :param input_dim: 输入特征的维数
        :param hidden_dim: LSTM单元内部隐藏状态维数
        :param num_layers: 叠的LSTM层的数量
        :param batch_first: (batch, seq, feature)
        """
        self.fc = nn.Linear(self.hidden_dim * self.seq_len, self.input_dim * self.pred_len)

    def forward(self, x):
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)  # [Batch_size, seq_len, d_model]
        lstm_out = lstm_out.contiguous().view(batch_size, -1)
        output = self.fc(lstm_out)  # [Batch_size, pred_len * Nodes]
        output = output.view(batch_size, self.pred_len, self.input_dim)
        return output, None
