# PC Softmax Higher Accuracy Demonstration (CUB-200-2011)

**Acknowledgment: This repository is based on the implementation by:**
[*Large Scale Fine-Grained Categorization  and Domain-Specific Transfer Learning*](https://arxiv.org/abs/1806.06193)\
[Yin Cui](http://www.cs.cornell.edu/~ycui/), [Yang Song](https://ai.google/research/people/author38270), [Chen Sun](http://chensun.me/), Andrew Howard, [Serge Belongie](http://blogs.cornell.edu/techfaculty/serge-belongie/)\
CVPR 2018

This repository contains implementation for demonstrating that the 
Probability-Correct (PC) softmax can achieve higher classification
accuracy than the traditional softmax. This is with the CUB-200-2011
dataset. For more detailed information, please refer to the paper. 

## Installation

Please install all the necessary python packages, which is easy
and intuitive if one uses a package management system like Conda.

Please download the CUB-200-2011 dataset: 
[CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). 
Please unzip this file in './data/cub_200'. So the existing
'train.txt' and 'val.txt' will split the CUB-200-2011 dataset into 
the imbalanced training and test datasets. If you wish to use the 
balanced CUB-200-2011, just replace 'train.txt' and 'val.txt' 
with the 'train.txt' and 'val.txt' in the './data/cub_200/balanced'
directory. 

Please download the [Inception-V3](https://drive.google.com/open?id=1EUNR4o77lNt0fN5Bi4lKxTZnFhghILRw)
pretrained model and put in './checkpoints/'. 

## Usage

Please refer to the bash files in './scripts'. For example, 
to run with backbone VGG, without ADL, one can type: 
```bash
bash scripts/run_vgg_no_ADL.sh
```

Training a model takes approximately 4 hours with two 
Nvidia 2080 Ti GPUs. Since the pretrained models are huge, we will release the pretrained models in the
form of Google Drive links imminently if the paper has been accepted. 
Currently we do not provide those due for the sake of anonymous reviewing 
purposes. 

For more information, please refer to the original README file. 

## License
[CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/)