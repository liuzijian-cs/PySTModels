import torch
import torch.nn as nn
from model.iTransformer.layers.AttentionLayers import FullAttention, MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, attention_model, d_model, d_ffn, dropout=0.1, activation="relu", output_attention=False):
        super(EncoderLayer, self).__init__()
        self.attention_model = attention_model
        self.output_attention = output_attention

        d_ffn = d_ffn or 4 * d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ffn, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ffn, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = torch.nn.functional.relu if activation == "relu" else torch.nn.functional.gelu

    def forward(self, x, attn_mask=None):
        out_x, attention_weights = self.attention_model(x, x, x, attn_mask)  # attention
        x = x + self.dropout(out_x)  # resnet
        out_x = x = self.norm1(x)
        out_x = self.dropout(self.activation(self.conv1(out_x.transpose(-1, 1))))
        out_x = self.dropout(self.conv2(out_x).transpose(-1, 1))
        return self.norm2(out_x + x), attention_weights


class Encoder(nn.Module):
    def __init__(self, encoder_model, d_model):
        super(Encoder, self).__init__()
        self.encoder_model = nn.ModuleList(encoder_model)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        attention_weights_list = []
        for layer in self.encoder_model:
            x, attention_weights = layer(x, attn_mask)
            attention_weights_list.append(attention_weights)
        x = self.norm1(x)
        return x, attention_weights_list


if __name__ == '__main__':
    print("Testing: ...")
    # Q = torch.rand(64, 96, 8, 128)  # (B, L, H, E)
    # K = torch.rand(64, 12, 8, 128)  # (B, S, H, E)
    # V = torch.rand(64, 12, 8, 128)  # (B, S, H, E)
    Q = torch.rand(64, 96, 1024)  # (B, L, H, E)
    K = torch.rand(64, 12, 1024)  # (B, S, H, E)
    V = torch.rand(64, 12, 1024)  # (B, S, H, E)

    test_FullAttention = Encoder(
        [EncoderLayer(
            attention_model=MultiHeadAttention(
                attention_model=FullAttention(),
                d_model=1024,
                n_head=8,
                dropout=0.1
            ),
            d_model=1024,
            d_ffn=2048,
            dropout=0.1,
            activation="relu"
        ) for i in range(10)],
        d_model=1024
    )

    output, attention = test_FullAttention(Q)  # (B, L, H, E), (B, H, L, S)

    print(output.shape)
