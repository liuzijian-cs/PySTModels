import argparse
import torch
import numpy as np
import os

# Data Providers:
from utils.dataProviders import PEMS, SST
# Task Makers:
from utils.taskMaker import TimeSeriesForecast
# Models:
from model.iTransformer import iTransformer
from model.iTransformer_official import iTransformer_official

# Global dictionary:
data_dict = {
    'PEMS': PEMS,
    'SST': SST,
}

model_dict = {
    'iTransformer': iTransformer,
    'iTransformer_official': iTransformer_official
}

task_dict = {
    'TimeSeriesForecast': TimeSeriesForecast,
}

# Main:
if __name__ == '__main__':
    # 1. Parser
    parser = argparse.ArgumentParser()  # Create the argument parser

    # 1.1 Basic config:
    # 1.1.1 Device config
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--gpu_id', type=str, default="0", help='GPU device id (single gpu)')
    parser.add_argument('--use_multi_gpu', type=bool, default=False, help='')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='GPU device id (multi gpu)')
    parser.add_argument('--num_workers', type=int, default=0, help='CPU workers, if Windows system == 0 !')
    parser.add_argument('--log_interval_iter', type=int, default=1, help='log output every number of iter')
    # 1.2.1 Base path config
    parser.add_argument('--model_save_path', type=str, default='./model_save')
    parser.add_argument('--log_file', type=str, default='./model_save/logs/logs.txt')

    parser.add_argument('--model', type=str, default='iTransformer_official',
                        help='model list: [iTransformer, iTransformer_official]')
    parser.add_argument('--task', type=str, default='TimeSeriesForecast',
                        help='task list:[Task]')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    # 1.2 Data arguments:
    # 1.2.1 Basic
    parser.add_argument('--data', type=str, default='SST',
                        help='data list: [PEMS, SST], new dataset pls conf in utils/dataset_conf')
    parser.add_argument('--data_path', type=str, default='./data/SST/Nan_Hai.csv')

    # 1.2.2 Forecasting Task
    parser.add_argument('--features', type=str, default='M',
                        help='features list: [M, S, MS], ''M: mul predict mul, S: uni pred uni , MS: mul pred uni')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    # encoder: |--------------- seq_len ---------------|
    # decoder: :                        |---label_len---|--pred_len--|
    parser.add_argument('--seq_len', type=int, default=96, help='Look back up sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='options')
    parser.add_argument('--pred_len', type=int, default=12, help='predict sequence length')

    # 1.2.3 Train/Val/Test ratio
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='valid ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio')

    # 1.2 Model arguments:
    parser.add_argument('--epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Based on the size of the GPU memory')
    parser.add_argument('--early_stopping', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--amp', type=bool, default=False, help='Automatic Mixed Precision')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle data')
    parser.add_argument('--scale', type=bool, default=True, help='Whether to normalize the original data')
    parser.add_argument('--scaler', type=str, default=None, help='Specify the scaler,None = std scaler')  # TODO
    parser.add_argument('--d_model', type=int, default=1024, help='Dimensionality of the model')  # TODO
    parser.add_argument('--d_ff', type=int, default=1024, help='Dimensionality of the FFN')
    parser.add_argument('--enc_layers', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=4, help='Number of decoder layers')
    # 1.2.1 Attention
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='dropout in attention modules')

    parser.add_argument('--channel_independence', type=bool, default=False, help='channel_independence')
    # Ablation experiment
    parser.add_argument('--output_attention', type=bool, default=False, help='return the attention matrix')

    # TimeSeries:
    parser.add_argument("--data_inverse_scale", type=bool, default=True, help='inverse')  # TODO

    # 1.2
    # 1.2 Model:
    # Common Parameter:

    args = parser.parse_args()
    logfile = open(args.log_file, "w")

    # 2. Initialization:
    # 2.3 Path existence checking # TODO
    # if not os.path.exists(args.save_dir):

    Task = task_dict[args.task].Task(args, model_dict, data_dict)
    Task.train()
