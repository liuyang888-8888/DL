# -*- coding: utf-8 -*-
#  道阻且长，行则将至
# -----Sunnyln---
#  2023/10/27  15:48
import torch
from  torch import nn
def comp_conv2d(conv2d,X):
    X=X.reshape((1,1)+X.shape)
    Y=conv2d(X)

    return Y.reshape(Y.shape[2:])

conv2d=nn.Conv2d(1,1,kernel_size=3,padding=1)
X=torch.rand(size=(8,8))
print(comp_conv2d(conv2d,X).shape)



conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print( comp_conv2d(conv2d, X).shape)


conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)