import random
import torch
import torch.utils.data as utils
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import StepLR

from utils import own_softmax
from model import ClassNet
import torch.nn.functional as F
import numpy as np
import math

import argparse

parser = argparse.ArgumentParser(description='Arguments for Softmax Mutual Information Estimator.')
parser.add_argument('--class-num', default=5, choices=[5],
                    help='Class Label numbers (Currently only support 5). ')
parser.add_argument('--data-dim', default=1, choices=[1, 2, 5, 10],
                    help='Data dimensionality. ')
parser.add_argument('--dataset-type', default='balanced', choices=['balanced', 'imbalanced'],
                    help='Dataset being balanced or imbalanced. ')
parser.add_argument('--pre-model', default='pretrained_models/balanced_dataset/class_net_best_lat_1.ckpt',
                    help='Path to a pretrained model state dict. ')
parser.add_argument('--pc-softmax', default=True, help='To use the probability correct softmax. ')
args = parser.parse_args()

class_num = args.class_num
data_dim = args.data_dim
data_bal = args.dataset_type
pre_model = args.pre_model
is_pc_softmax = args.pc_softmax

data_bal = 'balanced'
# pre_model = 'pretrained_models/balanced_dataset/class_net_best_lat_10.ckpt'
pre_model = '/media/zhenyue-qin/New Volume/Research/Research-Submissions/Submission-InfoCAM/Sayaka-InfoCAM/Research-Sayaka-InfoCAM/softmax_MI_estimator/class_net_best.ckpt'
is_pc_softmax = False
data_dim = 1

if data_bal == 'balanced':
    tra_each_data_set_num = [10000, 10000, 10000, 10000, 10000]
elif data_bal == 'imbalanced':
    tra_each_data_set_num = [5000, 10000, 15000, 20000, 25000]
else:
    raise AssertionError

# assert data_bal in pre_model

# Create distributions
normal_dists = []
mean_unit = 10
normal_dists.append(torch.distributions.MultivariateNormal(torch.zeros(data_dim), torch.eye(data_dim)))
for i in range(1, int(class_num / 2) + 1):
    normal_dists.append(torch.distributions.MultivariateNormal(torch.ones(data_dim) * i * mean_unit,
                                                               torch.eye(data_dim)))
    normal_dists.append(torch.distributions.MultivariateNormal(-torch.ones(data_dim) * i * mean_unit,
                                                               torch.eye(data_dim)))

tra_all_data = []
test_all_data = []
test_each_data_set_num = list([int(x / 5) for x in tra_each_data_set_num])
total_data_set_num = sum(tra_each_data_set_num)
dataset_prop = list([x / total_data_set_num for x in tra_each_data_set_num])

for i in range(max(tra_each_data_set_num)):
    for j in range(class_num):
        if i < tra_each_data_set_num[j]:
            tmp_ = normal_dists[j].sample([1])
            class_ = torch.ones((1, 1)) * j
            tmp_ = torch.cat((tmp_, class_), dim=-1)
            tra_all_data.append(tmp_)

for i in range(max(test_each_data_set_num)):
    for j in range(class_num):
        if i < test_each_data_set_num[j]:
            tmp_ = normal_dists[j].sample([1])
            class_ = torch.ones((1, 1)) * j
            tmp_ = torch.cat((tmp_, class_), dim=-1)
            test_all_data.append(tmp_)

tra_data = torch.cat(tra_all_data, dim=0)
test_data = torch.cat(test_all_data, dim=0)

in_tra_data = tra_data[:, :data_dim]
tra_label = tra_data[:, data_dim:].long()

tra_label = tra_label[torch.randperm(tra_label.shape[0])]

in_test_data = test_data[:, :data_dim]
test_label = test_data[:, data_dim:].long()

bch_sz = 2000
tra_dataset = utils.TensorDataset(in_tra_data, tra_label)
tra_dataloader = DataLoader(tra_dataset, num_workers=1, batch_size=bch_sz, shuffle=True, drop_last=False,
                            pin_memory=True)

test_dataset = utils.TensorDataset(in_test_data, test_label)
test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=bch_sz, shuffle=True, drop_last=False,
                            pin_memory=True)


class_net = ClassNet(data_dim, class_num, is_pc_softmax).to('cuda')
criterion = nn.NLLLoss()

# class_net.load_state_dict(torch.load(pre_model))

# optimizer = optim.Adam(class_net.parameters(), lr=1e-4)
optimizer = optim.SGD(class_net.parameters(), lr=5e-5)

def test():
    correct = 0
    total = 0
    total_loss = 0
    for ite_idx, (a_data, a_label) in enumerate(test_dataloader):
        a_data = a_data.to('cuda')
        a_label = a_label.to('cuda').squeeze()
        test_ = class_net(a_data, dataset_prop)
        loss = criterion(test_, a_label)
        _, predicted = torch.max(test_, 1)
        total += a_label.size(0)
        correct += (predicted == a_label).sum().item()
        total_loss += loss

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    return correct, total_loss


def train():
    max_accu = -1
    epoch_num = 150
    for epoch in range(1, epoch_num+1):
        correct = 0
        running_loss = 0.0
        sum_loss = 0
        for ite_idx, (a_data, a_label) in enumerate(tra_dataloader):
            a_data = a_data.to('cuda')
            a_label = a_label.to('cuda').squeeze()
            preds_ = class_net(a_data, dataset_prop)
            loss = criterion(preds_, a_label)
            loss.backward()
            optimizer.step()
            sum_loss += loss

            running_loss += loss.item()

        print('epoch: ', epoch)
        print('running loss: ', running_loss)
        _, predicted = torch.max(preds_, 1)
        correct += (predicted == a_label).sum().item()
        print('Accuracy of the network on the 10000 train images: %d %%' % (
                100 * correct / a_label.size(0)))

        test_acc, test_loss = test()

        if test_acc >= max_accu:
            min_loss = test_loss
            max_accu = test_acc
            torch.save(class_net.state_dict(), 'class_net_best.ckpt')

    torch.save(class_net.state_dict(), 'class_net.ckpt')


def evaluate():
    print('MI evaluation. ')
    softmax_list = []
    x_y_prob_list = []
    for a_data, a_label in test_dataset:
        a_data_org = a_data
        a_data = a_data.unsqueeze(0).to('cuda')
        a_label = a_label.to('cuda')
        int_label = a_label.cpu().item()
        dist_ = normal_dists[int_label]
        x_y_prob = torch.exp(dist_.log_prob(a_data_org))

        p_x_marg = 0
        dist_idx = 0
        for a_dist in normal_dists:
            p_x_marg += dataset_prop[dist_idx] * torch.exp(a_dist.log_prob(a_data_org))
            dist_idx += 1

        a_log_x_prob = x_y_prob / p_x_marg
        test_ = class_net(a_data, dataset_prop)
        if is_pc_softmax:
            a_softmax = own_softmax(test_, dataset_prop)[0][int_label]
        else:
            a_softmax = torch.softmax(test_, dim=-1)[0][int_label]
        # print('a softmax: ', a_softmax)

        if is_pc_softmax:
            softmax_list.append(math.log(a_softmax.cpu().item()))
        else:
            softmax_list.append(math.log(a_softmax.cpu().item()) + math.log(len(dataset_prop)))

        x_y_prob_list.append(math.log(a_log_x_prob.cpu().item()))

    print('x y prob: ', np.mean(x_y_prob_list))
    print('softmax: ', np.mean(softmax_list))


train()
test()
evaluate()


