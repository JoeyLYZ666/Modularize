import os

model = 'GCN'
dataset = 'R8'
# target_class from 0 to 7 (default: 0)
target_class = 0 

cmd = f'python -u grad_splitter.py --model {model} --dataset {dataset} ' \
      f'--target_class {target_class} > {model}_{dataset}_target_class_{target_class}.log'
print(cmd)
os.chdir('../')
os.system(cmd)