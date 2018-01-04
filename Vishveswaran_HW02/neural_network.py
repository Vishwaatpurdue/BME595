#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 23:10:33 2017

@author: vishwa
"""

import torch
#import numpy as np
from torch import np


class NeuralNetwork:
    # This class consists of neural network model which helps to solve the logic gates
    def __init__(self,*args):
        # This function initializes all the parameters needed for the neural network layer.
        # It initialises the no. of hidden layers in accordance with the input argument
        # It also initializes the size of hidden layer, input layer, output layer and the weights inclusive of the biases
        #self.ip_sz=args[0]
        self._op_sz=args[len(args)-1]
        self._theta={}
        self._theta_sz=len(args)
        for i in range(len(args)-1):
            # In this block the weights for the bias term is included
            self._theta[i]=torch.FloatTensor(torch.np.random.normal(loc=0,scale=(1/np.sqrt(args[i]+1)),size=(args[i]+1,args[i+1])))
            
        return
    def getlayer(self,layer):
        # This function is to provide the weights for each layer
        return self._theta[layer]
    def sigmoid(self,ip):
        # This function generates the sigmoid function replica of torch
        self._sig=1/(1+np.exp(-ip))
        return self._sig
    def forward(self,ip):
        # This function executes the forward propagation of the neural network
        # try catch is implemented to get whether it is 1D or 2D
        self._ip=ip
        try:
            self._chk=self._ip.size()[1]
            
            for i in range(self._theta_sz-1):
                self._lay=torch.matmul(self._theta[i].transpose(-2,1),self._ip)
                self._lay=self.sigmoid(self._lay)
                self._ip=torch.FloatTensor(torch.np.zeros((self._lay.size()[0]+1,self._lay.size()[1])))
                if self._lay.size()[0]==1:
                    self._ip[1,:]=self._lay
                else:
                    self._ip[1:,:]=self._lay
                self._ip[0,:]=1.0
            self._op=self._ip[1:,:]
            
        except IndexError:
            
            for i in range(self._theta_sz-1):
                self._lay=torch.matmul(self._theta[i].transpose(-2,1),self._ip)
                self._lay=self.sigmoid(self._lay)
                self._ip=torch.FloatTensor(torch.np.zeros(self._lay.size()[0]+1))
                if self._lay.size()[0]<2:
                    self._ip[1]=self._lay[0]
                else:
                    self._ip[1:]=self._lay
                self._ip[0]=1.0
            self._op=self._ip[1:]
        return self._op

#
#a=NeuralNetwork(1,1,1,1,2)
#i=torch.FloatTensor(np.array([[1.0,2.0],[1.0,2.0]]))
#i=torch.FloatTensor(np.array([1.0,2.0]))
#a=NeuralNetwork(2,2,2,2,2)
#i=torch.FloatTensor(np.array([[1.0,2.0],[1.0,2.0],[3.0,4.0]]))
#i=torch.FloatTensor(np.array([1.0,2.0,3.0]))
#d=a.forward(i)
#print(d)

