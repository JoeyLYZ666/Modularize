import os
import sys
sys.path.append('..')

# 1. intra-network reuse
# CIFAR
#F1最高的module的索引
module_idx = '6,1,10,6,3,3,10,6,1,1'  # SimCNN
# module_idx = '14,1,4,9,14,12,12,9,1,12'  # ResCNN
# module_idx = '4,1,10,9,5,9,9,5,12,12'  # InceCNN

# SVHN
# module_idx = '1,9,11,8,1,8,0,9,5,11'  # SimCNN
# module_idx = '6,5,5,0,2,6,0,0,5,11'  # ResCNN
# module_idx = '3,2,5,6,2,6,6,5,5,5'  # InceCNN

model = 'simcnn'
dataset = 'cifar10'
# Python 解释器运行脚本时使用“无缓冲”模式。具体来说，-u 参数是用来关闭标准输出的缓冲，以便立即将输出写入到控制台而不进行缓存。
cmd = f'python -u src/experiments/ensemble_modules.py ' \
      f'--model {model} --dataset {dataset} --estimator_indices {module_idx}'
print(cmd)
os.system(cmd)


# 2. inter-network reuse
# CIFAR
# module_idx = "'0,6|1,1|0,10|1,9|1,14|0,3|1,12|0,6|1,1|0,1'"  # SimCNN-ResCNN
# module_idx = "'2,4|0,1|0,10|2,9|2,5|2,9|2,9|0,6|2,12|0,1'"  # SimCNN-InceCNN
# module_idx = "'2,4|1,1|2,10|2,9|2,5|2,9|2,9|2,5|2,12|2,12'"  # ResCNN-InceCNN
# module_idx = "'2,4|1,1|0,10|2,9|2,5|2,9|2,9|0,6|2,12|0,1'"  # SimCNN-ResCNN-InceCNN

# SVHN
# module_idx = "'1,6|0,9|0,11|0,8|0,1|0,8|0,0|0,9|0,5|1,11'"  # SimCNN-ResCNN
# module_idx = "'0,1|0,9|0,11|0,8|0,1|0,8|0,0|0,9|0,5|2,5'"  # SimCNN-InceCNN
# module_idx = "'1,6|2,2|1,5|1,0|2,2|1,6|1,0|1,0|2,5|1,11'"  # ResCNN-InceCNN
# module_idx = "'1,6|0,9|0,11|0,8|0,1|0,8|0,0|0,9|0,5|1,11'"  # SimCNN-ResCNN-InceCNN

# dataset = 'svhn'
# cmd = f'python -u ../experiments/ensemble_modules_across_model.py ' \
#       f'--dataset {dataset} --estimator_indices {module_idx}'
# print(cmd)
# os.system(cmd)