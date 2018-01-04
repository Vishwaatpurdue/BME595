#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:05:16 2017

@author: vishwa
"""
from neural_network import NeuralNetwork
import numpy as np
import torch
class AND(NeuralNetwork):
    def __init__(self):
        # This function initializes the neural network for the AND gate
        NeuralNetwork.build(self,2,1)
        
        return
    def forward(self,inpbool):
        # Input is considered to be a list, hence converted into an array
        # This function accomodates both batch opertion as well as single vector operation, hence a try catch is used
        # This function calls the forward operation in the NeuralNetwork class to compute the forward propagation
        inpbool=np.asarray(inpbool)
        try:
            self.chk=np.size(inpbool,1)
            self.temp=torch.FloatTensor(torch.np.zeros((np.size(inpbool,0),np.size(inpbool,1))))
            # Boolean inputs are converted into float tensors
            for i in range(np.size(inpbool,0)):
                for j in range(np.size(inpbool,1)):
                    if(inpbool[i,j]==True):
                        self.temp[i,j]=1.0
                    else:
                        self.temp[i,j]=0.0
            self.inp=torch.cat((torch.FloatTensor(torch.np.ones((1,self.chk))),self.temp),0)        
            self.out=NeuralNetwork.forward(self,self.inp)
            self.final_op=[True if (j >=0.8) else False for i in self.out for j in i]    
        except IndexError:
            self.temp=torch.FloatTensor(np.asanyarray([1.0 if (i ==True) else 0.0 for i in inpbool ]))
            self.inp=torch.cat((torch.FloatTensor(torch.np.ones(1)),self.temp),0)        
            self.out=NeuralNetwork.forward(self,self.inp)
            # The float tensors output from forward propagation of NN is converted back into boolean
            self.final_op=[True if (i >=0.8) else False for i in self.out]
        return self.final_op
    def train(self):
        # This function trains the weights
        self.traindata=np.asarray([np.array([True,True,False,False]*100),np.array([True,False,True,False]*100)])
        self.target=torch.FloatTensor(np.array([1.0 if self.traindata[0,j] and self.traindata[1,j] else 0.0 for j in range(400)]))
        self.iter=400
        self.eta = 0.2
        for loop in range(self.iter):
            self.forward(self.traindata[:,loop])
            self.backward(self.target[loop],'CE')
            #self.forward(self.traindata)
            #self.backward(self.target)
            self.updateParams(self.eta)
        self.forward(self.traindata)
        
        print("The final weights after %d iterations with %0.01f are:"%(self.iter,self.eta))
        print(self.getlayer(0))
        print("The training accuracy is: %0.001f"%(100-(torch.sum((self.out-self.target)**2)/self.target.size()[0]*100)))
        
        return 

class OR(NeuralNetwork):
    def __init__(self):
        # This function initializes the neural network for the AND gate
        NeuralNetwork.build(self,2,1)
        
        return
    def forward(self,inpbool):
        # Input is considered to be a list, hence converted into an array
        # This function accomodates both batch opertion as well as single vector operation, hence a try catch is used
        # This function calls the forward operation in the NeuralNetwork class to compute the forward propagation
        inpbool=np.asarray(inpbool)
        try:
            self.chk=np.size(inpbool,1)
            self.temp=torch.FloatTensor(torch.np.zeros((np.size(inpbool,0),np.size(inpbool,1))))
            # Boolean inputs are converted into float tensors
            for i in range(np.size(inpbool,0)):
                for j in range(np.size(inpbool,1)):
                    if(inpbool[i,j]==True):
                        self.temp[i,j]=1.0
                    else:
                        self.temp[i,j]=0.0
            self.inp=torch.cat((torch.FloatTensor(torch.np.ones((1,self.chk))),self.temp),0)        
            self.out=NeuralNetwork.forward(self,self.inp)
            self.final_op=[True if (j >=0.8) else False for i in self.out for j in i]    
        except IndexError:
            self.temp=torch.FloatTensor(np.asanyarray([1.0 if (i ==True) else 0.0 for i in inpbool ]))
            self.inp=torch.cat((torch.FloatTensor(torch.np.ones(1)),self.temp),0)        
            self.out=NeuralNetwork.forward(self,self.inp)
            # The float tensors output from forward propagation of NN is converted back into boolean
            self.final_op=[True if (i >=0.8) else False for i in self.out]
        return self.final_op
    def train(self):
        # This function trains the weights
        self.traindata=np.asarray([np.array([True,True,False,False]*100),np.array([True,False,True,False]*100)])
        self.target=torch.FloatTensor(np.array([1.0 if self.traindata[0,j] or self.traindata[1,j] else 0.0 for j in range(400)]))
        self.iter=400
        self.eta = 0.7
        for loop in range(self.iter):
            #self.forward(self.traindata[:,loop])
            #self.backward(self.target[loop])
            self.forward(self.traindata)
            self.backward(self.target,'CE')
            self.updateParams(self.eta)
        self.forward(self.traindata)
        
        print("The final weights after %d iterations with %0.01f are:"%(self.iter,self.eta))
        print(self.getlayer(0))
        print("The training accuracy is: %0.001f"%(100-(torch.sum((self.out-self.target)**2)/self.target.size()[0]*100)))
        
        return 

class NOT(NeuralNetwork):
    def __init__(self):
        NeuralNetwork.build(self,1,1)
        # Get the layer and modify the values for the layer to suit the AND operation
        # The threshold for True or False is 0.5
        return
    def forward(self,inpbool):
        # Input is considered to be a list, hence converted into an array
        # This function accomodates both batch opertion as well as single vector operation, hence a try catch is used
        # This function calls the forward operation in the NeuralNetwork class to compute the forward propagation
        inpbool=np.asarray(inpbool)
        if(np.size(inpbool)>1):
            self.temp=torch.FloatTensor(np.asanyarray([1.0 if (i ==True) else 0.0 for i in inpbool ]))
            # Boolean inputs are converted into float tensors
            self.inp=torch.FloatTensor(torch.np.ones((2,np.size(inpbool))))
            self.inp[1,:]=self.temp
            self.out=NeuralNetwork.forward(self,self.inp)
            self.final_op=[True if (j >0.5) else False for i in self.out for j in i]    
        else:
            self.temp=torch.FloatTensor(np.asanyarray([1.0 if (inpbool ==True) else 0.0]))
            self.inp=torch.cat((torch.FloatTensor(torch.np.ones(1)),self.temp),0)        
            self.out=NeuralNetwork.forward(self,self.inp)
            # The float tensors output from forward propagation of NN is converted back into boolean
            self.final_op=[True if (i >0.5) else False for i in self.out]
        return self.final_op
    
    def train(self):
        # This function trains the weights
        self.traindata=np.array([True,False]*200)
        self.target=torch.FloatTensor(np.array([1.0 if not (self.traindata[j]) else 0.0 for j in range(400)]))
        self.iter=400
        self.eta = 0.1
        for loop in range(self.iter):
            self.forward(self.traindata[loop])
            self.backward(self.target[loop],'CE')
            #self.forward(self.traindata)
            #self.backward(self.target)
            self.updateParams(self.eta)
        self.forward(self.traindata)
        
        print("The final weights after %d iterations with %0.01f are:"%(self.iter,self.eta))
        print(self.getlayer(0))
        print("The training accuracy is: %0.001f"%(100-(torch.sum((self.out-self.target)**2)/self.target.size()[0]*100)))
        
        return 

class XOR(NeuralNetwork):
    def __init__(self):
        # This function initializes the neural network for the AND gate
        NeuralNetwork.build(self,2,2,1)
        
        return
    def forward(self,inpbool):
        # Input is considered to be a list, hence converted into an array
        # This function accomodates both batch opertion as well as single vector operation, hence a try catch is used
        # This function calls the forward operation in the NeuralNetwork class to compute the forward propagation
        inpbool=np.asarray(inpbool)
        try:
            self.chk=np.size(inpbool,1)
            self.temp=torch.FloatTensor(torch.np.zeros((np.size(inpbool,0),np.size(inpbool,1))))
            # Boolean inputs are converted into float tensors
            for i in range(np.size(inpbool,0)):
                for j in range(np.size(inpbool,1)):
                    if(inpbool[i,j]==True):
                        self.temp[i,j]=1.0
                    else:
                        self.temp[i,j]=0.0
            self.inp=torch.cat((torch.FloatTensor(torch.np.ones((1,self.chk))),self.temp),0)        
            self.out=NeuralNetwork.forward(self,self.inp)
            self.final_op=[True if (j >0.7) else False for i in self.out for j in i]    
        except IndexError:
            self.temp=torch.FloatTensor(np.asanyarray([1.0 if (i ==True) else 0.0 for i in inpbool ]))
            self.inp=torch.cat((torch.FloatTensor(torch.np.ones(1)),self.temp),0)        
            self.out=NeuralNetwork.forward(self,self.inp)
            # The float tensors output from forward propagation of NN is converted back into boolean
            self.final_op=[True if (i >0.7) else False for i in self.out]
        return self.final_op
    def train(self):
        # This function trains the weights
        self.traindata=np.asarray([np.array([True,True,False,False]*100),np.array([True,False,True,False]*100)])
        self.target=torch.FloatTensor(np.array([1.0 if ((self.traindata[0,j] or self.traindata[1,j]))and(not(self.traindata[0,j] and self.traindata[1,j])) else 0.0 for j in range(400)]))
        self.iter=400 # 4000 for batch optimization
        self.eta = 0.2
        for loop in range(self.iter):
            self.forward(self.traindata[:,loop])
            self.backward(self.target[loop],'CE')
            #self.forward(self.traindata)
            #self.backward(self.target,'CE')
            self.updateParams(self.eta)
        self.forward(self.traindata)
        
        print("The final weights after %d iterations with %0.01f are:"%(self.iter,self.eta))
        print(self.getlayer(0))
        print(self.getlayer(1))
        print("The training accuracy is: %0.001f"%(100-(torch.sum((self.out-self.target)**2)/self.target.size()[0]*100)))
        
        return 


