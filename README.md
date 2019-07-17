# Chinese-Fortune-Telling
A software about chinese fortune telling with Chinese name and face
## The state
We trained a face beauty detecting model with dataset； https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
## Requirement
Tensorflow (over 1.6), python 2.7
## Details
model: Resnet_v1_50


The model parameters are initialized by the pretrained CNN models of ImageNet and updated by mini-batch Stochastic Gardient Descent (SGD), where the learning rate is initialized as 0.001 and decreased by a factor of 10 per 5000 iterations. We set the batchsize as 16, momentum coefficient as 0.9. Weight decay coefficient as 1e-4 for ResNet.

## Run
Download dataset from https://pan.baidu.com/s/1Ff2W2VLJ1ZbWSeV5JbF0Iw(passwoard: if7p)

unzip it as 'data' fold.
### train
```Bash
python train.py
```
### test
```Bash
python test.py
```
## Result
MSE loss: 0.248 on train while 0.318 on test.

## Reference
https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
