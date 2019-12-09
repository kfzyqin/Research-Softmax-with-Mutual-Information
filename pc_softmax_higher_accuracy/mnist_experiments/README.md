# PC Softmax Higher Accuracy Demonstration (MNIST)

This repository contains implementation for demonstrating that the 
Probability-Correct (PC) softmax can achieve higher classification
accuracy than the traditional softmax. This is with the MNIST
dataset. For more detailed information, please refer to the paper. 

## Installation

Please install all the necessary python packages, which is easy
and intuitive if one uses a package management system like Conda.

## Usage

```bash
python classification_mnist.py
```

Please change `softmax_type` in `classification_mnist.py` as `True`
or `False` for using the pc-softmax or the traditional-softmax. 
This place is marked with comments `Switch these two`. 

Please change these lines in function `def make_custom_dataset(a_dataset, tgt_labels, to_train=True)`: 
```python
shorten_data_num = avg_data_num
long_data_num = avg_data_num
```
to change the imbalance of the dataset. 

We do not provide pretrained models since training can be done in 
a minute. 

## License
[CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/)