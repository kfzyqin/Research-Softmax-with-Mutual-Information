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

Please refer to the original README file. 

## Usage

Please refer to the original README file. 

The original splitting of the CUB-200-2011 dataset leads to a
balanced one. We include data_splits for creating an imbalanced
dataset. The files and tools are in the "data_splits" directory. 

Please change `logit_end_denom` in `train.py` for using the pc-softmax or the traditional-softmax. 
This place is marked with comments `Switch these two`. 

Training a model takes approximately 30 minutes with two 
Nvidia 2080 Ti GPUs. Since the pretrained models are huge, we will release the pretrained models in the
form of Google Drive links imminently if the paper has been accepted. 
Currently we do not provide those due for the sake of anonymous reviewing 
purposes. 


## License
[CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/)