#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:04:36 2017

@author: vishwa
"""

import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
class img2num(nn.Module):
    
    def __init__(self):
        # Initializes 3 layers of Linear layer(Logistic layer) with MSE as loss
        super(img2num, self).__init__() # calling parent class
        self.conv1 = nn.Conv2d(1, 6, 5,1,padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5,1)
        self.conv3 = nn.Conv2d(16,120,5)
        self.l1= nn.Linear(120, 84)
        self.l2= nn.Linear(84, 10)
        #self.loss_cric = nn.MSELoss()
        self.loss_cric = nn.CrossEntropyLoss()
        
        return
    def train(self):
        # This function trains the model developed using nn Module in pytorch 
        # for the MNIST dataset
        # Also, It initializes input data loader. Since cross validation or testing 
        # is not used, we did not load the test data set
        eta=0.2
        epochs=30
        self.batch_size=200
        #self.batch_size=1
        train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=self.batch_size, shuffle=True)
        optimizer=optim.SGD(self.parameters(),lr=eta)
        for epoch in range(epochs):
            #for batch_idx, (data, target) in enumerate(train_loader):
            for batchidx,(data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                data,target=Variable(data),Variable(target)
                #data=Variable(data)
                #target=Variable(self.__oneHot__(target))
                self.forward(data)
                oploss=self.loss_cric(self.out,target)                
                oploss.backward()
                optimizer.step()
            print("The error at each epoch is "+str(oploss.data[0])) 
        return
    
    def forward(self,ip):
        # This function provides forward propagation and identifies the desired number for that image
        if (isinstance(ip,torch.ByteTensor)):
            if(ip.dim()==2):
                temp=ip.numpy()
                temp=temp/255.0
                ip=np.zeros((1,1,28,28))
                ip[0,0,:,:]=temp
            else:
                temp=ip.numpy()
                ip=temp/255.0
            ip=Variable(torch.FloatTensor(ip))
        
        out=self.conv1(ip)
        out=F.max_pool2d(out,2,2)
        out=self.conv2(out)
        out=F.max_pool2d(out,2,2)
        out=self.conv3(out)
        out=out.view(ip.size()[0],-1)
        out=F.relu(self.l1(out))
        self.out=F.relu(self.l2(out))
        dummy,fout=torch.max(self.out,1)
        if len(fout)<2:
            return (fout.data)[0]
        else:
            return fout.data
    def __oneHot__(self,lbl):
        onehot=torch.FloatTensor(torch.zeros(10))
        onehot[lbl]=1.0
        return onehot
