from global_configure import GlobalConfigures


class Configures(GlobalConfigures):
    def __init__(self):
        super(Configures, self).__init__()
        self.model_name = 'GCN'
        self.dataset_name = 'R8'
        self.num_classes = 8
        self.num_conv = 2
        self.workspace = f'{self.data_dir}/{self.model_name}_{self.dataset_name}'
        self.set_model()
