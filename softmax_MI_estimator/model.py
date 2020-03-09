import torch
from torch import nn
import torch.nn.functional as F

from utils import own_softmax


class ClassNet(nn.Module):
    def __init__(self, in_dim, out_dim, to_emp_softmax=True):
        super(ClassNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, out_dim)

        self.to_emp_softmax = to_emp_softmax

    def forward(self, x_in, label_proportions):
        x_in = torch.relu(self.fc1(x_in))
        x_in = torch.relu(self.fc2(x_in))
        x_in = torch.relu(self.fc3(x_in))
        x_in = self.fc4(x_in)
        if label_proportions is not None and self.to_emp_softmax:
            x_in = torch.log(own_softmax(x_in, label_proportions) + 1e-5)
        else:
            x_in = torch.log(F.softmax(x_in, dim=1) + 1e-6)
        return x_in
