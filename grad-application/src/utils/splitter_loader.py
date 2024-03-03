import sys
sys.path.append('..')


def load_splitter(model_name, dataset_name):
    if dataset_name is None:
        GradSplitter = load_splitter_normal(model_name)
    else:
        raise ValueError()
    return GradSplitter


def load_splitter_normal(model_name):
    if model_name == 'GCN':
        from splitters.gcn_splitter import GradSplitter
    else:
        raise ValueError()
    return GradSplitter

