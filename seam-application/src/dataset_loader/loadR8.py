import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import numpy as np
from config import load_config

def extract_target_class(dataset, target_class, reorganize):
    tc_data_idx = np.where(np.array(dataset.targets) == target_class)[0]
    tc_data = dataset.data[tc_data_idx]
    if reorganize:
        tc_targets = [0] * len(tc_data)
    else:
        tc_targets = [target_class] * len(tc_data)

    non_tc_data_idx = np.where(np.array(dataset.targets) != target_class)[0]
    np.random.seed(0)
    np.random.shuffle(non_tc_data_idx)
    non_tc_data_idx = non_tc_data_idx[:len(tc_data)]
    non_tc_data = dataset.data[non_tc_data_idx]
    if reorganize:
        non_tc_targets = [1] * len(tc_data)
    else:
        non_tc_targets = [dataset.targets[i] for i in non_tc_data_idx]

    dataset.data = np.concatenate((tc_data, non_tc_data), axis=0)
    dataset.targets = tc_targets + non_tc_targets
    return dataset

def load_r8(is_train = False, target_class= 0, reorganize = False):
    config = load_config()
    r8_path = f'{config.dataset_dir}\\r8'
    r8_train_dir = f'{r8_path}\\train_txt'
    TRAIN_FILE = 'train_txt.txt'
    TRAIN_LABEL = 'train_label.txt'
    # The mean and std could be different in different developers;
    # however, this will not influence the test accuracy much.
    dtrain_set = DP.TxtDatasetProcessing(r8_path, r8_train_dir, TRAIN_FILE, TRAIN_LABEL, 32, corpus)

    dataset =  datasets.CIFAR10(f'{config.dataset_dir}', train=is_train, transform=transform)

    if target_class >= 0:
        assert 0 <= target_class <= 9
        dataset = extract_target_class(dataset, target_class, reorganize)

    return dataset


if __name__ == '__main__':
    load_r8(is_train=True, target_class=1, reorganize=True)

