import sys
sys.path.append('..')


def load_configure(model_name, dataset_name):
    model_dataset_name = f'{model_name}_{dataset_name}'
    if model_dataset_name == 'GCN_R8':
        from configures.gcn_r8 import Configures
    else:
        raise ValueError()
    configs = Configures()
    return configs
