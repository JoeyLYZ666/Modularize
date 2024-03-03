import os

# the seeds for randomly sampling from the original training dataset based on Dirichlet Distribution.
estimator_indices = [1, 3, 4, 6, 8, 10, 11, 14, 15, 16]

# parallel
model = 'simcnn'
dataset = 'cifar10'
#>:定向结果输出到log里面
for i, estimator_idx in enumerate(estimator_indices):
    cmd = f'python -u ../experiments/evaluate_modules.py ' \
          f'--model {model} --dataset {dataset} --estimator_idx {estimator_idx} ' \
          f'> ./eval_{model}_{dataset}_estimator_{estimator_idx}.log'
    print(cmd)
    os.system(cmd)


