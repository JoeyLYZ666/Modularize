# Modularize GCN through GradSplitter

## Introduction
With the widespread success of deep learning technologies, many trained deep neural network (DNN) models are now publicly available. However, directly reusing the public DNN models for new tasks often fails due to mismatching functionality or performance. Inspired by the manner named GradSplitter, our experiments use GradSplitter to modularize GCN on one widely-used public dataset into modules, each representing one of the N classes and containing a part of the convolution kernels of the trained CNN model.


## Requirements
+ argparse 1.4.0<br>
+ numpy 1.19.2<br>
+ python 3.8.10<br>
+ pytorch 1.8.1<br>
+ torchvision 0.9.0<br>
+ scikit-learn 0.22<br>
+ tqdm 4.61.0<br>
+ GPU with CUDA support is also needed


## How to install
Install the dependent packages via pip:

    $ pip install argparse==1.4.0 numpy==1.19.2 scikit-learn==0.22 tqdm==4.61.0
    
Install pytorch according to your environment, see [here](https://pytorch.org/.)



## Preparing
1. modify `root_dir` in `src/global_configure.py`.


## Modularizing GCN
1. run `python decompose.py` in `script/` (cd grad-application/src/scripts) to modularize the trained CNN model on target class which comes from 0 to 7.
3. The best module locates in grad-application\data\gcn_R8\module\class_n_lr_0.1_0.1_alpha_0.1\best\best.pth.(n refers to target_class such as class_0)


## Modular tools: GradSplitter
[GradSplitter](https://github.com/qibinhang/GradSplitter)


## Model
We use trained models from [here](https://github.com/zshicode/GNN-for-text-classification)


## Dataset
We use the following dataset for our example: [R8 Dataset](http://www.cs.umb.edu/~smimarog/textmining/datasets/).