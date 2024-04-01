import torch
import torch.nn as nn


class InvertedEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(InvertedEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mask):
        x = x.transpose(1, 2)  # x [B,T,N] -> x [B,N,T]
        if x_mask is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mask.transpose(1, 2)], dim=1))
        return self.dropout(x)