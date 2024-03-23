import argparse

import torch
import numpy as np

if __name__ == '__main__':

    # 1. Parser
    parser = argparse.ArgumentParser()  # Create the argument parser

    # 1.1 Basic config:
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
    parser.add_argument('--model', type=str, default='iTransformer', required=True,
                        help='model list: [iTransformer]')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    # 1.2 Model arguments:

    # 1.2
    # 1.2 Model:
    # Common Parameter:

    args = parser.parse_args()

    # Initialization:

    # Device initialization:
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(f'cuda is not available! (args.device == {args.deivce})')
    # Seed initialization:
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


