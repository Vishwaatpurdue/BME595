#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:04:36 2017

@author: vishwa
"""
import cv2
import torch
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
class img2obj(nn.Module):
    
    def __init__(self):
        # Initializes 3 layers of Linear layer(Logistic layer) with MSE as loss
        super(img2obj, self).__init__() # calling parent class
        self.conv1 = nn.Conv2d(3, 6, 5,1)
        self.conv2 = nn.Conv2d(6, 16, 5,1)
        self.conv3 = nn.Conv2d(16,480,5)
        self.l1= nn.Linear(480, 240)
        self.l2= nn.Linear(240, 100)
        self.smax=nn.LogSoftmax()
        #self.loss_cric = nn.MSELoss()
        #self.loss_cric = nn.CrossEntropyLoss()
        self.loss_cric = nn.NLLLoss()
        return
    def train(self):
        # This function trains the model developed using nn Module in pytorch 
        # for the MNIST dataset
        # Also, It initializes input data loader. Since cross validation or testing 
        # is not used, we did not load the test data set
        eta=0.05
        epochs=20
        self.batch_size=50
        #self.batch_size=1
        train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./cifardata', train=True, transform=transforms.ToTensor(), target_transform=None, download=True),
    batch_size=self.batch_size, shuffle=True)
        self.__unpickle__() # Load labels in labels variable
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
            print("The error at epoch "+str(epoch)+" is :"+str(oploss.data[0])) 
        return
    
    def forward(self,img):
        # This function provides forward propagation and identifies the desired number for that image
        if (isinstance(img,torch.ByteTensor)):
            if(img.dim()==3):
                temp=img.numpy()
                temp=temp/255.0
                img=np.zeros((1,3,32,32))
                img[0,:,:,:]=temp
            else:
                temp=img.numpy()
                img=temp/255.0
            img=Variable(torch.FloatTensor(img))
        if(isinstance(img,Variable)):
            if (img.dim()==3):
                temp=img.data
                img=torch.FloatTensor(np.zeros((1,3,32,32)))
                img[0]=temp
                img=Variable(img)
        out=self.conv1(img)
        out=F.max_pool2d(out,2,2)
        out=self.conv2(out)
        out=F.max_pool2d(out,2,2)
        out=self.conv3(out)
        out=out.view(img.size()[0],-1)
        out=F.relu(self.l1(out))
        self.out=self.smax(self.l2(out))
        dummy,fout=torch.max(self.out,1)
        if len(fout)<2:
            lbl=(fout.data)[0]
            return self.labels[lbl]
        else:
            return fout
        
    def view(self,img):
        # This function is to view the image and its corresponding caption 
        # deduced by the Lenet5 architecture using matplotlib
        
        caption=self.forward(img)
        if isinstance(img,Variable):
            img=img.data
        if img.dim()==4:
            img=img[0]
        img=transforms.ToPILImage()(img)       
        plt.figure()
        
        plt.title(caption)
        plt.imshow(img)
        plt.show()
        plt.pause(1)
        
        plt.close()
        
        return

    def cam(self,idx=0):
        # To get input images from live camera and caption it using deep learning
        self.train()
        cap = cv2.VideoCapture(idx)
        while(True):
            ret, frame = cap.read()
            frame=cv2.resize(frame,(32,32))
            frame_ip=torch.ByteTensor(frame)
            frame_ip=torch.transpose(frame_ip,0,2)
            caption=self.forward(frame_ip)
            print(caption)
            cv2.imshow('Output',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return
    def __oneHot__(self,lbl):
        onehot=torch.FloatTensor(torch.zeros(10))
        onehot[lbl]=1.0
        return onehot
    def __unpickle__(self):
        file='./cifardata/cifar-100-python/meta'
        with open(file, 'rb') as fo:
            self.labels = pickle.load(fo)
            self.labels=self.labels['fine_label_names']
        return
    
