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
    def build(self,*args):
        # This function initializes all the parameters needed for the neural network layer.
        # It initialises the no. of hidden layers in accordance with the input argument
        # It also initializes the size of hidden layer, input layer, output layer and the weights inclusive of the biases
        # This function just initializes theta, dE_dtheta and a private variable
        # act which stores the g(z) of each layer to compute back propagation
        self._theta={}
        self.__dE_dtheta={}
        self.__act={}
        self._op_sz=args[len(args)-1]
        self._theta_sz=len(args)-1
        for i in range(self._theta_sz):
            # In this block the weights for the bias term is included
            self._theta[i]=torch.FloatTensor(torch.np.random.normal(loc=0,scale=(1/np.sqrt(args[i]+1)),size=(args[i]+1,args[i+1])))
            self.__dE_dtheta[i]=torch.FloatTensor(torch.np.zeros((args[i]+1,args[i+1])))
        return
    def getlayer(self,layer):
        # This function is to provide the weights for each layer
        return self._theta[layer]
    def __sigmoid__(self,ip):
        # This function generates the sigmoid function replica of torch
        self._sig=1/(1+np.exp(-ip))
        return self._sig
    def forward(self,ip):
        # This function executes the forward propagation of the neural network
        # try catch is implemented to get whether it is 1D or 2D
        self._ip=ip
        try:
            self._chk=self._ip.size()[1]
            self.__act[0]=self._ip
            for i in range(self._theta_sz):
                self._lay=torch.matmul(self._theta[i].transpose(-2,1),self._ip)
                self._lay=self.__sigmoid__(self._lay)
                
                self._ip=torch.FloatTensor(torch.np.zeros((self._lay.size()[0]+1,self._lay.size()[1])))
                if self._lay.size()[0]==1:
                    self._ip[1,:]=self._lay
                else:
                    self._ip[1:,:]=self._lay
                self._ip[0,:]=1.0
                if (i==self._theta_sz-1):
                    self.__act[i+1]=self._lay
                else:
                    self.__act[i+1]=self._ip
            self.op=self._ip[1:,:]
            
        except IndexError:
            self.__act[0]=self._ip
            for i in range(self._theta_sz):
                self._lay=torch.matmul(self._theta[i].transpose(-2,1),self._ip)
                self._lay=self.__sigmoid__(self._lay)
                
                self._ip=torch.FloatTensor(torch.np.zeros(self._lay.size()[0]+1))
                if self._lay.size()[0]<2:
                    self._ip[1]=self._lay[0]
                else:
                    self._ip[1:]=self._lay
                self._ip[0]=1.0
                if (i==self._theta_sz-1):
                    self.__act[i+1]=self._lay
                else:
                    self.__act[i+1]=self._ip
            self.op=self._ip[1:]
        
        return self.op
    def backward(self,*args):
        # This function is to compute the back propagation for the neural network
        # using the target, which is the actual output 
        # args is implemented if the cross entropy is provided
        self.__target=args[0]
        try:
            self.CE=args[1]
        except IndexError:
            self.CE=None
        if(self.CE=='CE'):
            self.__err=torch.transpose(self.op-self.__target,0,-1)
            # Implementing SDG so we have to use each training data to compute the 
            # gradient and update the weights immediately and repeat the same to reach the consensus
            if(len(self.__err)<2):
                for i in range(self._theta_sz,0,-1):
                    if(i==self._theta_sz):
                        self.__layer_err=self.__err
                    else:
                        self.__temp=self.__act[i]*(1-self.__act[i])
                        self.__layer_err=self.__temp*self.__layer_err
                        self.__layer_err=self.__layer_err[1:]
                    for j in range(self.__dE_dtheta[i-1].size()[1]):
                        self.__dE_dtheta[i-1][:,j]=self.__layer_err[j]*(self.__act[i-1])
                    self.__layer_err=torch.matmul(self._theta[i-1],self.__layer_err)
            else:
                for item in range(len(self.__err)):
                # Now for each hidden layer we get
                    for i in range(self._theta_sz,0,-1):
                        if(i==self._theta_sz):
                            self.__layer_err=self.__err[item]
                        else:
                            self.__layer_err=self.__err[item]*self.__act[i][:,item]*(1-self.__act[i][:,item])
                            self.__layer_err=self.__layer_err[1:]
                        for j in range(self.__dE_dtheta[i-1].size()[1]):
                            self.__dE_dtheta[i-1][:,j]=self.__layer_err[j]*(self.__act[i-1][:,item])
                        self.__layer_err=torch.matmul(self._theta[i-1],self.__layer_err)
            for i in range(self._theta_sz):
                self.__dE_dtheta[i]=self.__dE_dtheta[i]/len(self.__err)    
            return
        # If cross entropy is not provided the following code executes
        self.__err=torch.transpose(self.op-self.__target,0,-1)
        
        # Implementing SDG so we have to use each training data to compute the 
        # gradient and update the weights immediately and repeat the same to reach the consensus
        if(len(self.__err)<2):
            for i in range(self._theta_sz,0,-1):
                self.__temp=self.__act[i]*(1-self.__act[i])
                
                if(i==self._theta_sz):
                    self.__layer_err=(self.__err*self.__temp)
                else:
                    self.__layer_err=self.__temp*self.__layer_err
                    self.__layer_err=self.__layer_err[1:]
                    
                for j in range(self.__dE_dtheta[i-1].size()[1]):
                    self.__dE_dtheta[i-1][:,j]=self.__layer_err[j]*(self.__act[i-1])
                self.__layer_err=torch.matmul(self._theta[i-1],self.__layer_err)
                
                #print(self.__layer_err)
        else:
            for item in range(len(self.__err)):
            # Now for each hidden layer we get
            
                for i in range(self._theta_sz,0,-1):
                    self.__temp=self.__act[i][:,item]*(1-self.__act[i][:,item])
                
                    if(i==self._theta_sz):
                        self.__layer_err=(self.__err[item]*self.__temp)
                    else:
                        self.__layer_err=self.__layer_err*self.__temp
                        #self.__layer_err=self.__layer_err[1:,:]
                        self.__layer_err=self.__layer_err[1:]
                    for j in range(self.__dE_dtheta[i-1].size()[1]):
                        self.__dE_dtheta[i-1][:,j]=self.__layer_err[j]*(self.__act[i-1][:,item])
                    self.__layer_err=torch.matmul(self._theta[i-1],self.__layer_err)
                    #print(self.__layer_err)
        for i in range(self._theta_sz):
            self.__dE_dtheta[i]=self.__dE_dtheta[i]/len(self.__err)
        return
    def updateParams(self,eta):
        # This function gets an input eta which the learning rate and updates the weights of the neural network
        self.eta=eta
        for i in range(self._theta_sz):
            self._theta[i]=self._theta[i]-(self.eta*self.__dE_dtheta[i])
        return

