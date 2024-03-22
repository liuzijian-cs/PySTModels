import argparse

import torch

if __name__ == '__main__':

    # 1. Parser
    parser = argparse.ArgumentParser()  # Create the argument parser

    # 1.1 Basic config:
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
    parser.add_argument('--model', type=str, default='iTransformer', required=True,
                        help='model list: [iTransformer]')

    # 1.2 Model arguments:


    # 1.2
    # 1.2 Model:
    ## Common Parameter:

    args = parser.parse_args()

    # init:

    # Device:
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print('No GPU found')
            args.device = 'cpu'
