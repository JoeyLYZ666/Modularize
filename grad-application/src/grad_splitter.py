from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F

import numpy as np
import shutil
import os
from utils.utils import *
from utils.gcn_loader import get_graph_inputs
from models.gcn import SimpleConv
from utils.gcn_loader import get_graph_inputs
import argparse
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from utils.model_loader import load_model
from utils.configure_loader import load_configure
from utils.checker import check_dir
from utils.splitter_loader import load_splitter


def loss_func(predicts, label, grad_splitter):
    module_params = grad_splitter.get_module_params()
    loss_pred = F.cross_entropy(predicts, label)

    #所有张量的元素求平均（m*1/n）
    loss_count = torch.mean(module_params)  # each module has fewer kernels
    return loss_pred, loss_count


def modularize_gcn(model, data , save_dir): # 8个model，每个model两个全连接层（每个全连接一对bias和weight）总共32个参数
    #默认初始化为1（mask）——包含10个模块
    grad_splitter = GradSplitter(model, init_type , data) # 8个模块（两层）

    acc_log = []
    best_acc, best_epoch, best_avg_kernel = 0.0, 0, 0
    best_modules = None
    #可更新参数中的head
    head_param = []
    # model 也有head但是他的head是不需要优化的
    for name , param in grad_splitter.named_parameters(): # 两个全连接层的weight和bias，共32
        if 'head' in name and param.requires_grad:
            head_param.append(param)
    all_param = [param for param in grad_splitter.parameters() if param.requires_grad]
    #当前轮数的策略:action[e]
    phase = 0
    #Adam优化器（优化head）
    optimize = torch.optim.Adam(head_param, lr=lr_head)
    ratio_loss_sim = 0

    for epoch in range(epochs_for_head + epochs_for_modularity):
        if phase != iterative_strategy[epoch]:
            phase = iterative_strategy[epoch]
            #初始的前几轮不用更新mask
            if phase == 0:
                print('\n*** Train head ***\n')
                optimize = torch.optim.Adam(head_param, lr=lr_head * 0.1)  # smaller lr_ratio
                ratio_loss_sim = 0
            else:
                print('\n*** Modularize ***\n')
                #训练head和mask
                optimize = torch.optim.Adam(all_param, lr=lr_modularity)
                ratio_loss_sim = alpha

        print(f'epoch {epoch}')
        print('-' * 80)
        #对于当前的输入数据训练一次并输出正确率以及PercentKernel（模型整体）
        grad_splitter, optimize = train_splitter_graph(grad_splitter, optimize, data, ratio_loss_sim)
        val_acc, avg_kernel = eval_splitter(grad_splitter, data)
        acc_log.append(val_acc)

        #记录最好的模块对应的epoch
        if val_acc >= best_acc:
            best_acc = val_acc
            best_avg_kernel = avg_kernel
            best_epoch = epoch
        #获得mask
        best_modules = grad_splitter.get_module_params()
        # DEMO\\grad-application/data/gcn_R8/module//class_0_lr_0.01_0.001_alpha_0.1
        torch.save(best_modules, f'{save_dir}/epoch_{epoch}.pth')
        
    print('='*100 + '\n')
    print(f'best_epoch: {best_epoch}')
    print(f'best_acc: {best_acc * 100:.2f}%')
    print(f'best_avg_kernel: {best_avg_kernel:.2f}')

    sorted_acc_log = list(sorted(zip(range(len(acc_log)), acc_log), key=lambda x: x[1], reverse=True))
    print(sorted_acc_log)

     # 源文件路径
    source_file = f'{configs.module_save_dir}/class_{args.target_class}_lr_{lr_head}_{lr_modularity}_alpha_{alpha}/epoch_{best_epoch}.pth'

    # 目标文件路径
    target_dir = f'{configs.module_save_dir}/class_{args.target_class}_lr_{lr_head}_{lr_modularity}_alpha_{alpha}/best/'
    target_file = f'{configs.module_save_dir}/class_{args.target_class}_lr_{lr_head}_{lr_modularity}_alpha_{alpha}/best/best.pth'
    os.makedirs(target_dir, exist_ok=True) 
    shutil.copy(source_file , target_file)
    return best_epoch


def train_splitter_graph(grad_splitter, optimize, data, ratio_loss_sim):
    # data = [g , adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size]
    epoch_train_loss_pred = []
    epoch_train_loss_sim = []
    epoch_train_acc = []
    grad_splitter.train()
    grad_splitter.model.eval()  # for BN in model（模型不改变，修改的只是mask）
   
    # inputs = [t_features , t_y_train , t_y_val , t_y_test , t_train_mask(训练部分的输出) , tm_train_mask , support , num_supports , t_support]
    inputs = get_graph_inputs(data[1], data[2], data[3], data[4], data[5], data[6])
    inputs[0] = inputs[0].to('cuda')
    grad_splitter = grad_splitter.to('cuda')
    logits = grad_splitter(inputs[0])
    inputs[1] = inputs[1].to('cuda')
    inputs[4] = inputs[4].to('cuda')
    inputs[5] = inputs[5].to('cuda')
    
    acc = ((torch.max(logits, 1)[1] == torch.max(inputs[1], 1)[1]).float() * inputs[4]).sum().item() / inputs[4].sum().item()        
    #之前的梯度信息清除
    optimize.zero_grad()
    #准确率：交叉熵，保留的核的平均数
    loss_pred, loss_sim = loss_func(logits * inputs[5], torch.max(inputs[1], 1)[1], grad_splitter)
    #ratio_loss_sim：保留的核的权重（不更新mask时候ratio_loss_sim是0）
    loss = loss_pred + ratio_loss_sim * loss_sim
    #计算的梯度信息存储在模型的参数张量
    loss.backward()

    grad_dict = {}
    # 在调用优化器的step之前，可以检查梯度：
    for name , param in grad_splitter.named_parameters():
        if param.requires_grad:
            grad_dict[name] = param.grad.cpu().detach().numpy()

    #执行优化—mask
    optimize.step()

    # pred = torch.argmax(logits * inputs[5], dim=1)
    # acc = torch.sum(pred == torch.max(inputs[1], 1)[1])
    #批处理的标签张量 batch_labels，它的形状为 (batch_size, ...)，batch_labels.shape[0]就是长度
    # epoch_train_acc.append(torch.div(acc, torch.max(inputs[1], 1)[1].shape[0]))
    epoch_train_acc.append(acc)
    #剔除计算图的信息也就是梯度信息
    epoch_train_loss_pred.append(loss_pred.detach())
    epoch_train_loss_sim.append(loss_sim.detach())

    print(f'## Train ##')
    print(f"loss_pred: {sum(epoch_train_loss_pred) / len(epoch_train_loss_pred):.2f}")
    print(f"loss_sim: {sum(epoch_train_loss_sim) / len(epoch_train_loss_sim):.2f}")
    print(f"acc : {sum(epoch_train_acc) / len(epoch_train_acc) * 100:.2f}%\n")
    return grad_splitter, optimize


@torch.no_grad()
def eval_splitter(grad_splitter, data, attr='val'):
    epoch_val_acc = []
    epoch_val_loss_pred = []
    epoch_val_loss_sim = []
    grad_splitter.eval()
    # data = [g , adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size]
    # inputs = [t_features , t_y_train , t_y_val , t_y_test , t_train_mask , tm_train_mask , support , num_supports , t_support]
    # t_features, t_y_test, test_mask
    inputs = get_graph_inputs(data[1], data[2], data[3], data[4], data[5], data[6])
    inputs[0] = inputs[0].to('cuda')
    grad_splitter = grad_splitter.to('cuda')
    logits = grad_splitter(inputs[0])
    t_mask = torch.from_numpy(np.array(data[8]*1., dtype=np.float32)).to(device)
    tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, inputs[3].shape[1])
    logits = logits.to(device)
    tm_mask = tm_mask.to(device)
    inputs[3] = inputs[3].to(device)
    #确保所有的都在cuda上面
    pred = torch.max(logits, 1)[1]
    acc = ((pred == torch.max(inputs[3], 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()
    loss_pred, loss_sim = loss_func(logits * tm_mask, torch.max(inputs[3], 1)[1], grad_splitter)

    epoch_val_acc.append(acc)
    epoch_val_loss_pred.append(loss_pred.detach())
    epoch_val_loss_sim.append(loss_sim.detach())
    val_acc = sum(epoch_val_acc) / len(epoch_val_acc)

    # n_kernel
    n_kernel = []
    #module_kernels：所有参数二值化
    module_params = grad_splitter.get_module_params() # 八个模块的weight
    #mask的阈值：0.5
    module_params = module_params > 0.5
    #每一个模块
    for module_idx in range(len(module_params)):
        #每一个模块的kernel数量
        n_kernel.append(torch.sum(module_params[module_idx]).float())
    #使用stack将张量堆叠在一起也就多一个维度即kernel的个数
    avg_kernel = torch.mean(torch.stack(n_kernel))

    if attr == 'val':
        print('## Validation ##')
    else:
        print('## Test ##')
    print(f"loss_pred: {sum(epoch_val_loss_pred) / len(epoch_val_loss_pred):.2f}")
    print(f"loss_sim: {sum(epoch_val_loss_sim) / len(epoch_val_loss_sim):.2f}")
    print(f"acc : {val_acc * 100:.2f}%\n")
    print(f'avg_kernels: {avg_kernel:.2f}\n')
    return val_acc, avg_kernel


def main():
    # 构造图
    print(torch.__version__)
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus('R8')
    features = sp.identity(features.shape[0])
    features = preprocess_features(features)
    g = construct_graph(adj)

    model = load_model(model_name=model_name , input_dim=features.shape[0], num_classes=y_train.shape[1] , g=g , conv = SimpleConv)
    #将模型参数加载进来，并加载参数到CUDA
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model.add_head() # head在加载参数之后加入
    #模型加载到CUDA
    model = model.to(device)
    #模型进入评估（不会进行梯度计算）
    model.eval()

    data = [g , adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size]

    if dataset_name == 'R8':
        # graph_loader = get_graph_loader(adj, features, y_train, y_val, y_test, train_mask)
        return modularize_gcn(model, data , module_save_dir)

    


if __name__ == '__main__':
    print('in')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn', 'incecnn' , 'GCN'] , default= 'GCN')
    parser.add_argument('--dataset', choices=['cifar10', 'svhn' , '20ng' , 'R8'] , default= 'R8')
    parser.add_argument('--target_class' , type = int , default = 0)
    parser.add_argument('--init_type', type=str, choices=['random', 'ones', 'zeros'], default='ones')
    parser.add_argument('--lr_head', type=float, default=0.1)
    parser.add_argument('--lr_modularity', type=float, default=0.1)
    parser.add_argument('--epoch_strategy', type=str, default='5,140', help='epochs_for_head,epochs_for_modularity')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.1)
    args = parser.parse_args()
    print(args)

    # 设置种子使得整个过程可以复现
    seed = 2019
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    #torch的生成随机数相对于numpy生成的随机数可能更加适合GPU进行操作
    torch.random.manual_seed(1)
    model_name = 'GCN'
    dataset_name = 'R8'
    # estimator_idx = 1
    init_type = args.init_type
    lr_head = args.lr_head
    lr_modularity = args.lr_modularity
    batch_size = args.batch_size
    epochs = args.epoch_strategy.split(',')
    #默认两个分别是5和140
    epochs_for_head, epochs_for_modularity = int(epochs[0]), int(epochs[1])
    #β（kernel保留率的权重）
    alpha = args.alpha
    #初始化GradSplitter——绑定model（SimCNN等）以及初始化卷积核参数+head（存储在param里面）
    GradSplitter = load_splitter(model_name, None)
    #训练策略：
    #epochs_for_head长度的全零数组
    iterative_for_head = [0] * epochs_for_head
    iterative_strategy = [1, 1, 1, 1, 1, 0, 0]  # 0 means training head, and 1 means modularization.
    iterative_strategy = iterative_strategy * int(epochs_for_modularity / 7)
    #最开始的5个左右由于head随机或者1初始化导致head的分类失误而非mask的原因，所以需要几个epoch来训练head
    #后续的策略是5个epoch训练mask，紧接着2个epoch训练head
    iterative_strategy = iterative_for_head + iterative_strategy

    #各种属性初始化
    configs = load_configure(model_name, dataset_name)
    dataset_dir = configs.dataset_dir
    trained_model_path = configs.trained_model_path
    module_save_dir = f'{configs.module_save_dir}/class_{args.target_class}_lr_{lr_head}_{lr_modularity}_alpha_{alpha}'
    check_dir(module_save_dir)
    main()
