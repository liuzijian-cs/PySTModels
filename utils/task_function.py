from utils.taskMaker.TimeSeriesForcast import TimeSeriesForcast
from utils.dataProviders import PEMS
from utils.base_function import print_log, Color
from model import iTransformer

data_dict = {
    'PEMS': PEMS,
}

model_dict = {
    'iTransformer': iTransformer
}

task_dict = {
    'TimeSeriesForcast': TimeSeriesForcast
}

if __name__ == '__main__':
    import argparse

    # 1. Parser
    parser = argparse.ArgumentParser()  # Create the argument parser

    # 1.1 Basic config:
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--gpu_id', type=str, default="0", help='GPU device id (single gpu)')
    parser.add_argument('--use_multi_gpu', type=bool, default=False, help='')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='GPU device id (multi gpu)')

    parser.add_argument('--num_workers', type=int, default=16, help='number of data loading workers')
    parser.add_argument('--model', type=str, default='iTransformer',
                        help='model list: [iTransformer]')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--model_save_path', type=str, default='../model_save')
    parser.add_argument('--log_file', type=str, default='../model_save/logs/logs.txt')

    # 1.2 Data arguments:
    # 1.2.1 Basic
    parser.add_argument('--data', type=str, default='PEMS',
                        help='data list: [PEMS], new dataset pls conf in utils/dataset_conf')
    parser.add_argument('--data_path', type=str, default='../data/PEMS/PEMS04.npz')

    # 1.2.2 Forecasting Task
    parser.add_argument('--features', type=str, default='M',
                        help='features list: [M, S, MS], ''M: mul predict mul, S: uni pred uni , MS: mul pred uni')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    # train (X):|--------------- seq_len ---------------|
    # label (Y):                        |---label_len---|--pred_len--|
    parser.add_argument('--seq_len', type=int, default=96, help='Look back up sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='options')
    parser.add_argument('--pred_len', type=int, default=12, help='predict sequence length')

    # 1.2.3 Train/Val/Test ratio
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='valid ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio')

    # 1.2 Model arguments:
    parser.add_argument('--epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Based on the size of the GPU memory')
    parser.add_argument('--early_stopping', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--amp', type=bool, default='True', help='Automatic Mixed Precision')

    # 1.2
    # 1.2 Model:
    # Common Parameter:

    args = parser.parse_args()

    print_log(args, 'task_function.py   :testing...')
    Task = TimeSeriesForcast(args, model_dict, data_dict)
