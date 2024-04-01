import numpy as np
import math
import torch
import torch.nn as nn


# AttentionLayers model list:
# |- FullAttention: (mask, scale, dropout, output_attention) forward(Q,K,V,attn_mask)
# |- MultiHeadAttention: (mask, scale, dropout, output_attention=False) forward(Q,K,V,attn_mask)


class FullAttention(nn.Module):
    def __init__(self, mask=False, scale=None, dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.mask = mask
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        scale = self.scale if self.scale is not None else 1. / math.sqrt(E)

        if self.mask:
            if attn_mask is None:
                with torch.no_grad():
                    attn_mask = torch.triu(torch.ones([B, 1, L, L], dtype=torch.bool), diagonal=1).to(queries.device)
            scores.masked_fill_(attn_mask, -float('inf'))
        attention_weights = torch.softmax(scores * 1. / scale, dim=-1)
        V = self.dropout(torch.einsum("bhls,bshe->blhe", attention_weights, values))
        return (V.contiguous(), None) if not self.output_attention else (V.contiguous(), attention_weights)


class MultiHeadAttention(nn.Module):
    """
    * EN: The input attention model can be adjusted and the QKV can be adjusted to a multi-head form.
    * CH:可以调整输入的attention模型，并将QKV调整为多头的形式
    """
    def __init__(self, attention_model, d_model, n_head, dropout=0.1, output_attention=False, d_key=None, d_value=None):
        super(MultiHeadAttention, self).__init__()
        self.attention_model = attention_model
        self.output_attention = output_attention
        self.d_model = d_model
        self.n_head = n_head

        d_key = d_key or d_model // n_head
        d_value = d_value or d_model // n_head

        self.query_linear = nn.Linear(d_model, d_key * n_head)
        self.key_linear = nn.Linear(d_model, d_key * n_head)
        self.value_linear = nn.Linear(d_model, d_value * n_head)
        self.out_projection = nn.Linear(d_value * n_head, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_head

        # The query, key, and value are linearly transformed and rearranged into a multi-header form
        # 对查询、键、值进行线性变换并重新排列为多头形式
        queries = self.query_linear(queries).view(B, L, H, -1)
        keys = self.key_linear(keys).view(B, S, H, -1)
        values = self.value_linear(values).view(B, S, H, -1)

        output, attention_weights = self.attention_model(queries, keys, values, attn_mask)
        return self.out_projection(output.view(B, L, -1)), attention_weights


if __name__ == '__main__':
    print("Testing: ...")
    # Q = torch.rand(64, 96, 8, 128)  # (B, L, H, E)
    # K = torch.rand(64, 12, 8, 128)  # (B, S, H, E)
    # V = torch.rand(64, 12, 8, 128)  # (B, S, H, E)
    Q = torch.rand(64, 96, 1024)  # (B, L, H, E)
    K = torch.rand(64, 12, 1024)  # (B, S, H, E)
    V = torch.rand(64, 12, 1024)  # (B, S, H, E)

    test_FullAttention = MultiHeadAttention(
        attention_model = FullAttention(),
        d_model = 1024,
        n_head = 8,
        dropout = 0.1
    )

    output, attention = test_FullAttention(Q, K, V, None)  # (B, L, H, E), (B, H, L, S)

    print(output.shape)
