from model import iTransformer
from utils.model_task.Model_TimeSeriesForcast import TimeSeriesForcast

model_dict = {
    'iTransformer': iTransformer
}

task_dict = {
    'TimeSeriesForcast': TimeSeriesForcast
}


def model_loader(args):
    None
