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
class NnImg2Num(nn.Module):
    
    def __init__(self):
        # Initializes 3 layers of Linear layer(Logistic layer) with MSE/CE as loss
        
        super(NnImg2Num, self).__init__() # calling parent class
        self.l1= nn.Linear(784, 500,bias=True)
        self.l2= nn.Linear(500, 150,bias=True)
        self.l3= nn.Linear(150, 10,bias=True)
        #self.loss_cric = nn.MSELoss()
        self.loss_cric = nn.CrossEntropyLoss()
        
        return
    def train(self):
        # This function trains the model developed using nn Module in pytorch 
        # for the MNIST dataset
        # Also It initialize input data loader, Since cross validation or testing 
        # is not used we did not load the test data set
        eta=0.5
        epochs=1
        batch_size=200
        train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
        optimizer=optim.SGD(self.parameters(),lr=eta)
        for epoch in range(epochs):
            #for batch_idx, (data, target) in enumerate(train_loader):
            for batchidx,(data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                #data,target=Variable(data),Variable(target)
                data=Variable(data)
                target=Variable(target)
                #target=Variable(self.__oneHot__(target))
                self.forward(data)
                oploss=self.loss_cric(self.out,target)                
                oploss.backward()
                optimizer.step()
            print("The error at epoch "+str(epoch)+" is :"+str(oploss.data[0])) 
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

        ip=ip.view(-1,784)
        out=F.relu(self.l1(ip))
        out=F.relu(self.l2(out))
        self.out=F.relu(self.l3(out))
        dummy,fout=torch.max(self.out,1)
        if len(fout)<2:
            return (fout.data)[0]
        else:
            return fout
    def __oneHot__(self,lbl):
        onehot=torch.FloatTensor(torch.zeros(10))
        onehot[lbl]=1.0
        return onehot
    
    