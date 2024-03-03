import sys
sys.path.append('..')
import loadR8


def load_dataset(dataset_name, is_train, shots=-1, target_class: int=-1, reorganize=False):
    if dataset_name == 'r8':
        dataset = load_r8(is_train, target_class=target_class, reorganize=reorganize)
    elif dataset_name == 'cifar100':
        dataset = load_cifar100(is_train, target_class=target_class, reorganize=reorganize)
    else:
        raise ValueError
    return dataset
