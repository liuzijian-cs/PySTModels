import torch
import torch.nn as nn


# Transformer: model & layer list:
# |- Model: Transformer model for Time serious forecasting.

class Model(nn.Module):
    def __init__(self, configs):
        """
        :param configs: {pred_len, output_attention, channel_independence, enc_in, dec_in, c_out, }
        """
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.ouput_attention = configs.ouput_attention

        # Decide whether the model should process each input channel independently
        # 决定模型是否应该独立地处理每个输入通道
        if configs.channel_independence:
            self.enc_in = 1
            self.enc_out = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.enc_out = configs.dec_in
            self.c_out = configs.c_out

        # Embedding
        # self.encoder =


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm_layer = norm_layer

    # def forward(self, x, attn_mask=None, tau=None, delta=None):


# class EncoderLayer(nn.Module):
#     None







