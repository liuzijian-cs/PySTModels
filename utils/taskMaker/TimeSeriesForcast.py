from .BasicTaskMaker import BasicTask


class TimeSeriesForcast(BasicTask):
    def __init__(self, args, model_dict, data_dict):
        super().__init__(args, model_dict, data_dict)