import os


def check_dir(target_dir):
    # GradSplitter-main\data\incecnn_cifar10\trained_models
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
