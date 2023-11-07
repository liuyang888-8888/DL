# -*- coding: utf-8 -*-
#  道阻且张，行则将至
# -----Sunnyln---
#  2023/9/15  17:34
import os

# os.makedirs(os.path.join('data'),exist_ok=True)
# data_file=os.path.join('data','house_tiny.csv')
# with open(data_file,'w') as f:
#     f.write('NumRooms,Alley,Price\n') #列名
#     f.write('NA,PAVE,127500\n')
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')

import pandas as pd
data=pd.read_csv('./data/house_tiny.csv')
inputs,outputs=data.iloc[:,0:2],data.iloc[:,2]
print(inputs)
inputs=inputs.fillna(inputs.select_dtypes(include='number').mean())
print(inputs)
inputs=pd.get_dummies(inputs,dummy_na=True)
print(inputs)
import torch
X=torch.tensor(inputs.to_numpy(dtype=float))
y=torch.tensor(outputs.to_numpy(dtype=float))
print(X,"\n",y)