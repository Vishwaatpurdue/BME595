#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:16:36 2017

@author: vishwa
"""

import torch
from torchvision import datasets, transforms
import numpy as np
import timeit
from torch.autograd import Variable
from img2num import img2num

#################################
#Testing - Part A
Nn=img2num()
start_time=timeit.default_timer()
Nn.train()
time_used=timeit.default_timer()-start_time
print("The time taken for training is "+str(time_used))
train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),batch_size=200)
f_err=0
# Inference
#def oneHot(lbl):
#        onehot=torch.FloatTensor(torch.zeros(10))
#        onehot[lbl]=1.0
#        return onehot
start_time=timeit.default_timer()
for idx,(data,target) in enumerate(train_loader): 
    data=Variable(data,volatile=True)
    target=Variable(target)
    #target=Variable(oneHot(target))
    op=Nn.forward(data)
    tgt1=(target.data).numpy()
    op1=(op).numpy()
    comp=np.asarray([1 if tgt1[i]!=op1[i] else 0 for i in range(200)]) #200
    f_err=f_err+np.sum(comp)
time_used=timeit.default_timer()-start_time
print("The total Error after training is "+str(f_err))
print("The time taken for inference is "+str(time_used))

a=np.zeros((28,28),dtype='uint8')
a[:,5:7]=1
a=torch.ByteTensor(a)
op=Nn.forward(a)