# Chinese-Fortune-Telling
A software about chinese fortune telling with someone's Chinese name and face
## The state
We trained a face beauty detecting model with dataset: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
## Requirement
Tensorflow (over 1.6), python 2.7
## Details
model: Resnet_v1_50


The model parameters are initialized by the pretrained CNN models of ImageNet and updated by mini-batch Stochastic Gardient Descent (SGD), where the learning rate is initialized as 0.001 and decreased by a factor of 10 per 5000 iterations. We set the batchsize as 16, momentum coefficient as 0.9. Weight decay coefficient as 1e-4 for ResNet.

## Run
Download dataset from https://pan.baidu.com/s/1Ff2W2VLJ1ZbWSeV5JbF0Iw(passwoard: if7p)

Unzip it as the 'data' fold.

Download pretrained model(Resnet_v1_50) from slim:https://github.com/tensorflow/models/tree/master/research/slim

And put it in the fold 'tmp_data'.
### train
```Bash
python train.py
```
### test
```Bash
python test.py
```
### face beauty detection(simple test)
| images      | score(1-5: higher score means someone is more beautiful.)     | 
| ---------- | :-----------:  | 
|   ![img](https://github.com/2012013382/Chinese-Fortune-Telling/blob/master/data/2.jpg)   | 3.67     | 
|   ![img](https://github.com/2012013382/Chinese-Fortune-Telling/blob/master/data/3.jpg)   | 2.20     | 
|   ![img](https://github.com/2012013382/Chinese-Fortune-Telling/blob/master/data/4.jpg)   | 3.00     | 
|   ![img](https://github.com/2012013382/Chinese-Fortune-Telling/blob/master/data/5.jpg)   | 3.08     | 
|   ![img](https://github.com/2012013382/Chinese-Fortune-Telling/blob/master/data/6.jpg)   | 1.82     | 
## Result
MSE loss: 0.248 on train while 0.318 on test.

## Reference
https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
