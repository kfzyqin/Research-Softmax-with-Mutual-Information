from __future__ import print_function
import torch
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import torch.utils.data as utils

import matplotlib.pyplot as plt

import numpy as np

bch_sz = 64

mnist_data = datasets.MNIST('.data',
                            train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))


def make_custom_dataset(a_dataset, tgt_labels, to_train=True):
    avg_data_num = 3000
    shorten_data_num = avg_data_num
    long_data_num = avg_data_num

    new_dataset = []
    new_labels = []

    if to_train:
        the_range = range(0, 40000)
    else:
        the_range = range(40000, 50000)
    label_count = {}
    for data_idx in the_range:
        a_data_label = a_dataset[data_idx]
        a_label = a_data_label[1]

        if a_label in tgt_labels:
            if a_label not in label_count.keys():
                label_count[a_label] = 1
            else:
                label_count[a_label] += 1
            if label_count[a_label] <= shorten_data_num:
                a_data = np.expand_dims(a_data_label[0], axis=0)
                new_dataset.append(a_data)
                new_labels.append(a_label)
        else:
            if a_label not in label_count.keys():
                label_count[a_label] = 1
            else:
                label_count[a_label] += 1
            if label_count[a_label] <= long_data_num:
                a_data = np.expand_dims(a_data_label[0], axis=0)
                new_dataset.append(a_data)
                new_labels.append(a_label)

    new_dataset = np.concatenate(new_dataset, axis=0)
    new_labels = np.stack(new_labels)

    new_dataset = torch.tensor(new_dataset)
    new_labels = torch.tensor(new_labels)
    my_dataset = utils.TensorDataset(new_dataset, new_labels)
    return my_dataset


tgt_fewer_labels = [0, 2, 4, 6, 8]
custom_train_dataset = make_custom_dataset(mnist_data, tgt_fewer_labels, to_train=True)
custom_test_dataset = make_custom_dataset(mnist_data, tgt_fewer_labels, to_train=False)

train_loader = DataLoader(custom_train_dataset, num_workers=1, batch_size=bch_sz, shuffle=True, drop_last=False,
                          pin_memory=True)
test_loader = DataLoader(custom_test_dataset, num_workers=1, batch_size=bch_sz, shuffle=True, drop_last=False,
                         pin_memory=True)

