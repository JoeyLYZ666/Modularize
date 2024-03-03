import os
import sys
sys.path.append('..')
from utils.configure_loader import load_configure

# After modularization, select the modules.
# First, considering the accuracy, the loss of accuracy should less than 1%.
# Then, considering the number of kernels.

best_epoch = [133, 111, 133, 111, 124, 99, 144, 114, 78, 86]

model = 'gcn'
dataset = 'r8'
lr_head = 0.01
lr_modularity = 0.001
alpha = 0.1  # for the weighted sum of loss1 and loss2
batch_size = 64

configs = load_configure(model, dataset)

for i, epoch in enumerate(best_epoch):
    #module_save_dir：data\modules\estimator_1\lr_0.01_0.001_alpha_0.1
    module_save_dir = f'{configs.module_save_dir}/lr_{lr_head}_{lr_modularity}_alpha_{alpha}'

    cmd = f'cp {module_save_dir}/epoch_{epoch}.pth ' 
    os.system(cmd)
    print(cmd)
