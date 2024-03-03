class GlobalConfigures:
    def __init__(self):
        #init函数里面的参数生申明了就是初始化
        import os
        root_dir = f'D:\\Lab\\BUAA\\DEMO\\grad-application'  # modify the dir
        assert os.path.exists(root_dir)

        if os.path.exists(f'{root_dir}/data'):
            self.data_dir = f'{root_dir}/data'
        else:
            raise ValueError(f'{root_dir}/data does not exist.')
        self.dataset_dir = f'{self.data_dir}/corpus'
        self.trained_entire_model_name = 'entire_model.pth'
        self.estimator_idx = None

        # Set in directory configures/
        self.workspace = None # workspace:'D:\\Lab\\BUAA\\DEMO\\grad-application/data/gcn_R8'
        #每次epoch处理的样本数量
        self.best_batch_size = None
        #CNNSplitter里面才需要α的值(α*ACC+(1-α)*Diff)
        self.best_alpha = None
        #最佳学习率
        self.best_lr = None
        self.best_epoch = None

        # define after setting estimator_idx
        self.trained_model_dir = None
        self.trained_model_path = None
        self.module_save_dir = None
        self.best_module_path = None

    def set_estimator_idx(self, idx):
        self.estimator_idx = idx
        self.trained_model_dir = f'{self.workspace}/trained_models'
        self.trained_model_path = f'{self.trained_model_dir}/estimator_{self.estimator_idx}.pth'
        #最好的模块的路径
        self.module_save_dir = f'{self.workspace}/modules/estimator_{self.estimator_idx}'
        self.best_module_path = f'{self.module_save_dir}/estimator_{self.estimator_idx}.pth'


    def set_model(self):
        self.trained_model_dir = 'D:\\Lab\\BUAA\\Demo\\data\\models'
        self.trained_model_path = f'{self.trained_model_dir}/GCN_R8.pt'
        #最好的模块的路径
        self.module_save_dir = f'{self.workspace}/module/'
        # self.best_module_path = f'{self.module_save_dir}/class_{args.target_class}.pth'