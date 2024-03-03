# Modularize LSTM through SEAM
## Abstract
With the widespread success of deep learning technologies, many trained deep neural network (DNN) models are now publicly available. However, directly reusing the public DNN models for new tasks often fails due to mismatching functionality or performance. Inspired by the manner named SEAM, our experiments use SEAM to modularize LSTM on one widely-used public dataset into modules, each representing one of the N classes and containing a part of the convolution kernels of the trained CNN model.


## Requirements
+ advertorch 0.2.3<br>
+ fvcore 0.1.5.post20220512<br>
+ matplotlib 3.4.2<br>
+ numpy 1.19.2<br>
+ python 3.8.10<br>
+ pytorch 1.8.1<br>
+ torchvision 0.9.0<br>
+ tqdm 4.61.0<br>
+ GPU with CUDA support is also needed

<br>

## Structure of the directories

```powershell
  |--- README.md                :  user guidance
  |--- global_config.py         :  setting the path
  |--- data                     :  the best module on target class
  |--- src/                     :  source code of our work
  |------ main.py               :  re-engineering a trained model and then reuse the re-engineered model
  |------ reengineering.py      :  reenegineer a trained model              
  |------ models/               :  trained models
  |------ utils/                :  tokenizer
  |------ scripts/              
  |------ ......
```

<br>


## Downloading model and dataset
1. The trained models from [here](https://github.com/jiangqy/LSTM-Classification-pytorch?tab=readme-ov-file)
2. The [R8 Dataset](http://www.cs.umb.edu/~smimarog/textmining/datasets/) used in the experimentss. <br>


## Preparing
Modify `self.root_dir` in `src/global_config.py`.


## Modularize the model on binary classification problems
1. cd seam-application/src/scripts
2. run the script:
```commandline
python model_reengineering.py --model lstm --dataset r8 --target_class 0 --lr_mask 0.1 --alpha 0.2
```
3. The modules are in data\binary_classification\lstm_r8\tr_n(n refers to target 0, such as tc_0)

## Modularize tool
See [SEAM](https://github.com/qibinhang/SeaM)