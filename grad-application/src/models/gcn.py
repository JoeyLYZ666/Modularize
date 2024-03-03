from __future__ import division
from __future__ import print_function
import torch
import math
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


device = torch.device('cuda')

class SimpleConv(nn.Module):
    def __init__(self,g,in_feats,out_feats,activation,feat_drop=True):
        super(SimpleConv, self).__init__()
        self.graph = g
        self.activation = activation
        setattr(self, 'W', nn.Parameter(torch.randn(in_feats,out_feats))) 
        self = self.to(device)
        self.feat_drop = feat_drop
    
    def forward(self, feat):
        # feat = feat.to('cuda')
        g = self.graph.to('cuda') 
        # feat = feat.to('cuda')  
        g.ndata['h'] = feat.mm(self.W) # 输入特征和矩阵相乘的结果存储到图节点的h中
        # fn.src_mul_edge(src='h', edge='w', out='m') 定义了消息传递函数，用于将源节点特征 'h' 乘邻接关系'w' 得到消息 'm'
        # fn.sum(msg='m',out='h') 定义了消息聚合函数
        g.update_all(fn.src_mul_edge(src='h', edge='w', out='m'), fn.sum(msg='m',out='h')) # 图消息传播
        rst = g.ndata['h']
        rst = self.activation(rst)
        return rst

class Classifer(nn.Module):
    def __init__(self,g,input_dim,num_classes,conv):
        super(Classifer, self).__init__()
        self.GCN = conv
        self.gcn1 = self.GCN(g,input_dim, 200, F.relu) # input_dim = 15362 , 200
        self.gcn2 = self.GCN(g, 200, num_classes, F.relu) # num_classes = 8
        self.num_classes = num_classes
        self.is_modular = True
    

    def add_head(self): # 8 to 2
        if self.is_modular:
            module_head_dim = 1
            self.module_head = nn.Sequential(
                nn.Linear(self.num_classes, 8),
                nn.ReLU(),
                nn.Linear(8, module_head_dim),
            )

    def forward(self, features):
        x = self.gcn1(features)
        self.embedding = x
        x = self.gcn2(x)
        return x
