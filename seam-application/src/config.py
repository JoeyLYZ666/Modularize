import sys
sys.path.append('seam-application')

from global_config import GlobalConfig


def load_config():
    config = Config()
    return config


class Config(GlobalConfig):
    def __init__(self):
        super().__init__()
        #project_data_save_dir:data/binary_classification
        self.project_data_save_dir = f'{self.data_dir}/binary_classification'
        
