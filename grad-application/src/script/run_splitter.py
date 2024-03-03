import os

# the seeds for randomly sampling from the original training dataset based on Dirichlet Distribution.
estimator_indices = [1, 3, 4, 6, 8, 10, 11, 14, 15, 16]

model = 'gcn'
dataset = '20ng'

for i, estimator_idx in enumerate(estimator_indices):
    cmd = f'python -u src/grad_splitter.py --model {model} --dataset {dataset} ' \
          f'--estimator_idx {estimator_idx} > {model}_{dataset}_estimator_{estimator_idx}.log'
    print(cmd)
    os.system(cmd)

