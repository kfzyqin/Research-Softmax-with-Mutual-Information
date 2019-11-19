# InfoCAM: Informative Class Activation Map (Tiny-ImageNet)

**Acknowledgment: This repository is based on the implementation by:**
[ADL: Attention-based Dropout Layer for Weakly Supervised Object Localization
](https://github.com/junsukchoe/ADL/tree/master/Pytorch)\
Junsuk Choe, Hyunjung Shim, CVPR 2018

This repository contains implementation for infoCAM: Informative
Class Activation Map. This is with the Tiny-ImageNet
dataset. For more detailed information, please refer to the paper. 

## Installation

Please install all the necessary python packages, which is easy
and intuitive if one uses a package management system like Conda.

Please download the Tiny-ImageNet dataset: 
[TinyImageNet](https://tiny-imagenet.herokuapp.com/). 
Please unzip this file in '../tiny-imagenet'.

Please download the pretrained [ResNet50](https://drive.google.com/open?id=0B7fNdx_jAqhtbllXbWxMVEdZclE)
pretrained model and put in './pretrained/'. 

## Usage

Please change `logit_end_denom` in `train.py` for using the pc-softmax or the traditional-softmax. 
This place is marked with comments `Switch these two`. 

(From the original README)
+ Convert dataset into '.tfrecord':
```
python convert_dataset.py --dataset_name=cub_200 --num_shards=10
```
+ Train (fine-tune) the model on 1 GPU:
```
CUDA_VISIBLE_DEVICES=0 ./train.sh
```
+ Evaluate the model on another GPU simultaneously:
```
CUDA_VISIBLE_DEVICES=1 ./eval.sh
```
+ Run Tensorboard for visualization:
```
tensorboard --logdir=./checkpoints/cub_200/ --port=6006
```

Training a model takes approximately 30 minutes with two 
Nvidia 2080 Ti GPUs. Since the pretrained models are huge, we will release the pretrained models in the
form of Google Drive links imminently if the paper has been accepted. 
Currently we do not provide those due for the sake of anonymous reviewing 
purposes. 

For more information, please refer to the original README file. 

## License
[CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/)