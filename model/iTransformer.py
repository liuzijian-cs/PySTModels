import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.seq_len = args.seq_len
        self.pren_len = args.pred_len


