import torch
import torch.nn as nn


class DataEmbeddingInverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        :param c_in:  seq_len
        :param d_model:
        :param embed_type:
        :param freq:
        :param dropout:
        """
        super(DataEmbeddingInverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mask):
        x = x.permute(0, 2, 1)  # [Batch, Variate(Nodes), Time] -> [Batch, Time, Variate(Nodes)] X^{N x T} -> X^{T x N}
        if x_mask is None:
            x = self.value_embedding(x)
