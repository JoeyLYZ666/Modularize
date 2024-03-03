from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from sklearn import metrics
import random
import time
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dgl
import dgl.function as fn
from dgl import DGLGraph
import numpy as np

from utils.utils import *

import argparse


def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Use CUDA training.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Initial learning rate.')
    parser.add_argument('--model', type=str, default="GCN",
                        choices=["GCN", "SAGE", "GAT"],
                        help='model to use.')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='require early stopping.')
    parser.add_argument('--dataset', type=str, default='20ng',
                        choices = ['20ng', 'R8', 'R52', 'ohsumed', 'mr'],
                        help='dataset to train')

    args, _ = parser.parse_known_args()
    #args.cuda = not args.no_cuda and th.cuda.is_available()
    return args

args = get_citation_args()

device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')


class GraphDataset(Dataset):
    def __init__(self , features , labels , masks , adjacent_matrix):
        self.features = features
        self.labels = labels
        self.masks = masks
        self.adjacent_matrix = adjacent_matrix
    
    
def get_graph_inputs(adj, features, y_train, y_val, y_test, train_mask): # 数据集loader
    # adjdense = torch.from_numpy(pre_adj(adj).A.astype(np.float32))
    # Define placeholders   
    '''
        1. features: array
        2. astype: type trasformation
        3. from_numpy: throw array to tensor
    '''
    t_features = torch.from_numpy(features.astype(np.float32))
    t_y_train = torch.from_numpy(y_train)
    t_y_val = torch.from_numpy(y_val)
    t_y_test = torch.from_numpy(y_test)
    # train_mask: Identify which ones belong to the training set
    t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
    '''
        1. torch.unsqueeze(t_train_mask, 0) add a new dimension
        2. torch.transpose(... , 1 , 0) changes the sort of dimension
        3. repeat(1 , y_train.shape[1])repeats the tensor itself to make sure the dimension ia the same as y_train        
    '''
    tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])
    support = [preprocess_adj(adj)]
    num_supports = 1
    t_support = [] # t_support：Adjacency matrix to tuple list and then to tensor list
    for i in range(len(support)):
        t_support.append(torch.Tensor(support[i]))

    inputs = [t_features , t_y_train , t_y_val , t_y_test , t_train_mask , tm_train_mask , support , num_supports , t_support]

    # graph_dataset = GraphDataset(t_features, [t_y_train, t_y_val, t_y_test], [t_train_mask, tm_train_mask], support)
    # graph_loader = DataLoader(graph_dataset, batch_size=1, shuffle=True)
    return inputs

