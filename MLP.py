# -*- coding: utf-8 -*-
#  道阻且张，行则将至
# -----Sunnyln---
#  2023/10/17  10:23
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs,num_outputs,num_hiddens = 784,10,256

W1 = nn.Parameter(torch.randn( num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn( num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]