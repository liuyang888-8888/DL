# -*- coding: utf-8 -*-
#  道阻且张，行则将至
# -----Sunnyln---
#  2023/9/27  18:05

import torch
from torch.distributions import multinomial
from matplotlib import pyplot as plt
fair_probs=torch.ones([6])/6
counts=multinomial.Multinomial(10,fair_probs).sample((1000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
plt.figure(figsize=(16,8))

for i in range(6):
    plt.plot(estimates[:,i].numpy(),label="P(die="+str(i+1)+")")
plt.axhline(y=0.167,color="black",linestyle="dashed")
plt.legend()
plt.show()