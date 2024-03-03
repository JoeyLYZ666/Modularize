# Modularize LSTM and GCN through GRadSplitter and SEAM
## Abstract
With the widespread success of deep learning technologies, many trained deep neural network (DNN) models are now publicly available. However, directly reusing the public DNN models for new tasks often fails due to mismatching functionality or performance. Inspired by the manner named SEAM and GradSplitter, our experiments use SEAM and GradSplitter to modularize LSTM and GCN on one widely-used public dataset into modules, each representing one of the N classes and containing a part of the convolution kernels(GCN) or weights(LSTM) of the trained models.

## Structure of the directory
```powershelll
  |--- README.md                  :  user guidance
  |--- data/                      :  trained models, datasets and modules
  |--- grad-application/          :  modularize GCN on R8 through GradSplitter
  |--- seam-application/          :  modularize LSTM on R8 through SEAM
  |--- other\                     :  The model training code pulled from Github and the processed dataset (specific URLs can be found in README.md in both project directories)
```
<br>

