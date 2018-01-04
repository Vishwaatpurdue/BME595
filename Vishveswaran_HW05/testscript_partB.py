#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 02:48:30 2017

@author: vishwa
"""

import torch
import numpy as np
import timeit
from torchvision import datasets, transforms
from torch.autograd import Variable
from img2obj import img2obj
#########################################################
# Testing

c_far=img2obj()
start_time=timeit.default_timer()
c_far.train()
time_used=timeit.default_timer()-start_time
print("The time taken for training is "+str(time_used))
train_loader = torch.utils.data.DataLoader(datasets.CIFAR100('./cifardata', train=True, transform=transforms.ToTensor(), target_transform=None, download=False),batch_size=100, shuffle=True)
f_err=0
# Inference
start_time=timeit.default_timer()
for idx,(data,target) in enumerate(train_loader): 
    data=Variable(data,volatile=True)
    target=Variable(target)
    #target=Variable(oneHot(target))
    op=c_far.forward(data)
    tgt1=(target.data).numpy()
    op1=(op.data).numpy()
    comp=np.asarray([1 if tgt1[i]!=op1[i] else 0 for i in range(100)]) #200
    f_err=f_err+np.sum(comp)
time_used=timeit.default_timer()-start_time
print("The total Error after training is "+str(f_err))
print("The time taken for inference is "+str(time_used))

train_loader = torch.utils.data.DataLoader(datasets.CIFAR100('./cifardata', train=False, transform=transforms.ToTensor(), target_transform=None, download=False),batch_size=100, shuffle=True)
test_err=0
# Testing model with testdata
start_time=timeit.default_timer()
for idx,(data,target) in enumerate(train_loader): 
    data=Variable(data,volatile=True)
    target=Variable(target)
    op=c_far.forward(data)
    tgt1=(target.data).numpy()
    
    op1=(op.data).numpy()
    comp=np.asarray([1 if tgt1[i]!=op1[i] else 0 for i in range(100)]) #200
    test_err=test_err+np.sum(comp)
time_used=timeit.default_timer()-start_time
print("The total Error after testing is "+str(test_err))
print("The time taken for inference is "+str(time_used))
c_far.view(data[-1])


########################################################
## Testing Real time camera
c_far=img2obj()
c_far.cam()