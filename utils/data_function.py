from torch.utils.data import DataLoader
from utils.dataProvider.PEMS import PEMS
from utils.dataset_conf.DataPEMS import DataPEMS
from utils.base_function import print_log

# function list:
# |-

data_dict = {
    'PEMS': DataPEMS,
    'PEMS_test': PEMS,
}


def data_loader(args, data_type):
    """
    :param args:
    :param data_type: ['train', 'valid', 'test']
    :return: dataset 、 dataloader
    """
    assert data_type in ['train', 'valid', 'test']
    # TODO: timeenc = 0 if args.embed != 'timeF' else 1
    Dataset = data_dict[args.data]

    conf_dict = {
        'batch_size': None,
        'shuffle': None,
        'num_workers': args.num_workers,
        'drop_last': False
    }

    if data_type == 'train' or 'valid':
        conf_dict['batch_size'] = args.batch_size
        conf_dict['shuffle'] = True
    elif data_type == 'test':
        conf_dict['batch_size'] = 1
        conf_dict['shuffle'] = False

    dataset = Dataset(args, data_type)
    print_log(args, f'data_function.py: Load dataset {args.data} , data type: {data_type}, size: {len(dataset)}')
    dataloader = DataLoader(dataset, **conf_dict)
    return dataset, dataloader

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='test dataProvider function')
    parser.add_argument('--log_file', type=str, default='../model_save/logs/logs.txt')
    parser.add_argument('--data', type=str, default='PEMS_test',
                        help='data list: [PEMS], new dataset pls conf in utils/dataset_conf')
    parser.add_argument('--data_path', type=str, default='../data/PEMS/PEMS04.npz')
    args = parser.parse_args()


    print_log(args,'data_function.py   :testing...')
    PEMS_train = data_dict[args.data](args)
