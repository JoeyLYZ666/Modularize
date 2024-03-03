class GlobalConfig:
    def __init__(self):
        self.root_dir = 'seam-application'
        self.data_dir = f'{self.root_dir}/data'
        self.dataset_dir = f'{self.data_dir}/dataset'
        print(self.dataset_dir)