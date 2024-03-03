import os

model = 'lstm'
dataset = 'r8'
# target_class from 0 to 7 (default: 0)
target_class = 0 

# cmd_0 = f'cd ../../..' # this command cannot change the curdir(each os.system() makes a new shell)
cmd_1 = f'python -u seam-application/src/main.py --model {model} --dataset {dataset} ' \
      f'--target_class {target_class} > {model}_{dataset}_target_class_{target_class}.log'
# print(cmd_0)
# os.system(cmd_0)
os.chdir('../../..')
print(os.curdir)
print(cmd_1)
os.system(cmd_1)