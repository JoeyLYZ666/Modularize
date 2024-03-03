from __future__ import division
from __future__ import print_function
import torch
import copy
import itertools
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from dgl import DGLGraph
import numpy as np

from utils.utils import *

import argparse


def get_dataset_loader(dataset_name, for_modular=False):
    if dataset_name == 'cifar10':
        #迪利克雷概率来抽样
        #返回的是函数的引用，并未调用
        load_dataset = _load_cifar10_dirichlet
    elif dataset_name == 'svhn':
        load_dataset = _load_svhn_dirichlet
    else:
        raise ValueError
    #这个函数供外部调用，所以封装内部调用函数在这里面
    return load_dataset
   


#函数名首位是下划线的代表默认在函数内部调用
def _load_cifar10_dirichlet(dataset_dir, is_train, shuffle_seed, is_random, split_train_set='8:2',
                            batch_size=64, num_workers=0, pin_memory=False):
    """
    shuffle_seed: the idx of a base estimator in the ensemble model.
    split_train_set: 8 for train model, 2 for validation of modularization. then 6 for train model, 2 for val model
    """
    #设置标准
    #三个颜色通道对应三个均值以及标准差
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #首先将图像的大小设置为32*32
    #转换为张量tensor
    #归一化
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    normalize])
    if is_train:
        #数据加强
        if is_random:
            transform = transforms.Compose([transforms.Resize((32, 32)),
                                            #随机像素水平翻转（概率有默认）
                                            #数据增强：如果图像是猫的左脸那么学习到的特征很可能是侧脸这个性质，狗的左脸也可能被识别
                                            transforms.RandomHorizontalFlip(),
                                            #随机剪裁，4表示在边界填充像素避免裁剪之后物体太过于靠近边界
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            normalize])
        #将split_train_set里面的比如8:2转换到[0.8 , 0.2]
        split_ratio = [int(i) / 10 for i in split_train_set.split(':')]
        assert sum(split_ratio) == 1
        #用 PyTorch 的 torchvision.datasets.CIFAR10 类加载 CIFAR-10 数据集的训练集。
        #root 参数指定数据集的根目录，train=True 表示加载训练集，transform 参数是前面定义的图像转换操作
        #加载训练集并进行数据的tranform处理
        train = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, transform=transform)

        alpha = 1
        #targets：十分类的类标签
        #十个类别一共至少抽取100个(CIFAR-10的训练集)
        sampled_indices = _dirichlet_sample(train.targets,
                                            n_classes=10, shuffle_seed=shuffle_seed, min_size=100, alpha=alpha)

        # split the indices of train set into 2 parts, including modularity_train_set and modularity_val_set
        #len(sampled_indices)：基于数据量分割
        train_set = sampled_indices[: int(split_ratio[0] * len(sampled_indices))]
        val_set = sampled_indices[int(split_ratio[0] * len(sampled_indices)):]
        split_set_indices = [train_set, val_set]

        # split the train set according the split indices.
        #训练集划分为子集，并根据子集创建DataLoad对象再进行训练
        split_set_loader = []
        #一共两次迭代分别是train_set,val_set
        for each_set_indices in split_set_indices:
            #训练子集深克隆（不影响原来的数据集）
            each_set = copy.deepcopy(train)
            #获取数据集中每个数据对应的标签
            each_set.targets = [each_set.targets[idx] for idx in each_set_indices]
            #给定数据索引获取数据并存储
            each_set.data = each_set.data[each_set_indices]
            #batch_size:小样本的个数64，num_workers：并行加载数据的进程个数，pin_memory：是否加载到内存
            #小样本数量：each_set大小/batch_size
            each_set_loader = DataLoader(each_set, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers, pin_memory=pin_memory)
            split_set_loader.append(each_set_loader)
        #返回数据子集（两个：训练和评估）对应的dataloader
        return split_set_loader
    else:
        ratio = 0.2  # 20% test data are used to evaluate modules.
        test = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, transform=transform)
        #数据长度的列表比如数据长度是10，那么total_indices是【0-9】
        total_indices = list(range(len(test)))
        #0.2验证，0.8测试(0-1999,2000-10000)
        module_eval_set, test_set = total_indices[:int(ratio * len(test))], total_indices[int(ratio * len(test)):]
        split_set_indices = [module_eval_set, test_set]
        # split the train set according the split indices.
        split_set_loader = []
        for each_set_indices in split_set_indices:
            each_set = copy.deepcopy(test)
            #每一条数据的target
            each_set.targets = [each_set.targets[idx] for idx in each_set_indices]
            each_set.data = each_set.data[each_set_indices]
            each_set_loader = DataLoader(each_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
            split_set_loader.append(each_set_loader)
        return split_set_loader


def _load_svhn_dirichlet(dataset_dir, is_train, shuffle_seed, is_random, split_train_set='8:2',
                         batch_size=64, num_workers=0, pin_memory=False):
    """
        shuffle_seed: the idx of a base estimator in the ensemble model.
        split_train_set: 8 for train model, 2 for validation of modularization. then 6 for train model, 2 for val model
        """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(),
                                    normalize])
    if is_train:
        if is_random:
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            normalize])
        split_ratio = [int(i) / 10 for i in split_train_set.split(':')]
        assert sum(split_ratio) == 1
        train = torchvision.datasets.SVHN(root=dataset_dir, split='train', transform=transform)

        alpha = 0.5
        sampled_indices = _dirichlet_sample(train.labels,
                                            n_classes=10, shuffle_seed=shuffle_seed, min_size=10, alpha=alpha)

        # split the indices of train set into 2 parts, including modularity_train_set and modularity_val_set
        train_set = sampled_indices[: int(split_ratio[0] * len(sampled_indices))]
        val_set = sampled_indices[int(split_ratio[0] * len(sampled_indices)):]
        split_set_indices = [train_set, val_set]

        # split the train set according the split indices.
        split_set_loader = []
        for each_set_indices in split_set_indices:
            each_set = copy.deepcopy(train)
            each_set.labels = each_set.labels[each_set_indices]
            each_set.data = each_set.data[each_set_indices]
            each_set_loader = DataLoader(each_set, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers, pin_memory=pin_memory)
            split_set_loader.append(each_set_loader)
        return split_set_loader
    else:
        ratio = 0.2  # 20% test data are used to evaluate modules.
        test = torchvision.datasets.SVHN(root=dataset_dir, split='test', transform=transform)
        total_indices = list(range(len(test)))
        module_eval_set, test_set = total_indices[:int(ratio * len(test))], total_indices[int(ratio * len(test)):]
        split_set_indices = [module_eval_set, test_set]
        # split the train set according the split indices.
        split_set_loader = []
        for each_set_indices in split_set_indices:
            each_set = copy.deepcopy(test)
            each_set.labels = each_set.labels[each_set_indices]
            each_set.data = each_set.data[each_set_indices]
            each_set_loader = DataLoader(each_set, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory)
            split_set_loader.append(each_set_loader)
        return split_set_loader


def _dirichlet_sample(dataset_labels, n_classes, shuffle_seed, min_size, alpha):
    #通过种子生成伪随机数，一样的数生成的伪随机数是一样的所以需要记录种子传入的顺序
    #使得后面的Proportions可以复现
    np.random.seed(shuffle_seed)
    while True:
        #np.repeat(alpha, n_classes):10个0.5的数组
        #获得迪利克雷分布的样本
        #np.random.dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, n_classes))
        #归一化
        proportions = proportions / np.max(proportions)
        data_idx_per_class = []
        for each_class in range(n_classes):
            #从类别0-9
            #找到label匹配的索引（where里面是条件）
            #np.where返回的是元组（包含多个数组的数据结构，但是这里只会有一个数组，[0]就是获取里面的第一个数组）
            #target_data_idx：每个label对应的数据索引
            target_data_idx = np.where(np.array(dataset_labels) == each_class)[0]
            target_data_idx = target_data_idx.tolist()
            np.random.shuffle(target_data_idx)
            #某个类别的比例
            ratio = proportions[each_class]
            #获取计算得到比例个元素（带冒号：切片操作）
            #加入切片（相当于一个数组）
            #data_idx_per_class：每个类别的数据
            data_idx_per_class.append(target_data_idx[:int(ratio * len(target_data_idx))])
        #如果类别的数量最小的低于10那就继续抽取数据指导每个类别最少有10个样本
        if min([len(each_class_sample) for each_class_sample in data_idx_per_class]) < min_size:
            continue
        else:
            break

    samples_idx = list(itertools.chain(*data_idx_per_class))
    np.random.shuffle(samples_idx)
    return samples_idx
