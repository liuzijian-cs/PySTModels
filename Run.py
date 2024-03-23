import argparse

import torch
import numpy as np

if __name__ == '__main__':

    # 1. Parser
    parser = argparse.ArgumentParser()  # Create the argument parser

    # 1.1 Basic config:
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
    parser.add_argument('--model', type=str, default='iTransformer',
                        help='model list: [iTransformer]')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    # 1.2 Data arguments:
    # 1.2.1 Basic
    parser.add_argument('--data', type=str, default='PEMS',
                        help='data list: [PEMS], new dataset pls conf in utils/dataset_conf')
    parser.add_argument('--data_path',type=str,default='./dataset/PEMS/PEMS04.npz')

    # 1.2.2 Forecasting Task
    parser.add_argument('--features', type=str, default='M',
                        hfelp='features list: [M, S, MS], ''M: mul predict mul, S: uni pred uni , MS: mul pred uni')
    parser.add_argument('--target', type=str, default='OT',help='target feature in S or MS task')
    # train (X):|--------------- seq_len ---------------|
    # label (Y):                        |---label_len---|--pred_len--|
    parser.add_argument('--seq_len', type=int, default=96, help='Look back up sequence length')
    parser.add_argument('--label_len', type=int, default=48,help='options')
    parser.add_argument('--pred_len', type=int, default=12,help='predict sequence length')

    # 1.2.3 Train/Val/Test ratio
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='valid ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio')

    # 1.2 Model arguments:




    # 1.2
    # 1.2 Model:
    # Common Parameter:

    args = parser.parse_args()

    # 2. Initialization:

    # 2.1 Device initialization:
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(f'cuda is not available! (args.device == {args.deivce})')
    # 2.2 Seed initialization:
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    # 2.3 Data initialization:







