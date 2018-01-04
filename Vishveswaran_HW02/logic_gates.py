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
        NeuralNetwork.__init__(self,2,1)
        # Get the layer and modify the values for the layer to suit the AND operation
        # The threshold for True or False is 0.5
        self.layer=NeuralNetwork.getlayer(self,0)
        self.layer[0]=-10
        self.layer[1]=6
        self.layer[2]=6
        return
    def __call__(self,inpbool):
        # Input is considered to be a list, hence converted into an array
        # This function accomodates both batch opertion as well as single vector operation, hence a try catch is used
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
            self.out=self.forward()
            self.final_op=[True if (j >0.5) else False for i in self.out for j in i]    
        except IndexError:
            self.temp=torch.FloatTensor(np.asanyarray([1.0 if (i ==True) else 0.0 for i in inpbool ]))
            self.inp=torch.cat((torch.FloatTensor(torch.np.ones(1)),self.temp),0)        
            self.out=self.forward()
            # The float tensors output from forward propagation of NN is converted back into boolean
            self.final_op=[True if (i >0.5) else False for i in self.out]
        return self.final_op
    def forward(self):
        # This function calls the forward operation in the NeuralNetwork class to compute the forward propagation
        return NeuralNetwork.forward(self,self.inp)

class OR(NeuralNetwork):
    def __init__(self):
        NeuralNetwork.__init__(self,2,1)
        # Get the layer and modify the values for the layer to suit the AND operation
        # The threshold for True or False is 0.5
        self.layer=NeuralNetwork.getlayer(self,0)
        self.layer[0]=-5
        self.layer[1]=6
        self.layer[2]=6
        return
    def __call__(self,inpbool):
        # Input is considered to be a list, hence converted into an array
        # This function accomodates both batch opertion as well as single vector operation, hence a try catch is used
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
            self.out=self.forward()
            self.final_op=[True if (j >0.5) else False for i in self.out for j in i]    
        except IndexError:
            self.temp=torch.FloatTensor(np.asanyarray([1.0 if (i ==True) else 0.0 for i in inpbool ]))
            self.inp=torch.cat((torch.FloatTensor(torch.np.ones(1)),self.temp),0)        
            self.out=self.forward()
            # The float tensors output from forward propagation of NN is converted back into boolean
            self.final_op=[True if (i >0.5) else False for i in self.out]
        return self.final_op
    def forward(self):
        # This function calls the forward operation in the NeuralNetwork class to compute the forward propagation
        return NeuralNetwork.forward(self,self.inp)

class NOT(NeuralNetwork):
    def __init__(self):
        NeuralNetwork.__init__(self,1,1)
        # Get the layer and modify the values for the layer to suit the AND operation
        # The threshold for True or False is 0.5
        self.layer=NeuralNetwork.getlayer(self,0)
        self.layer[0]=5
        self.layer[1]=-6
        return
    def __call__(self,inpbool):
        # Input is considered to be a list, hence converted into an array
        # This function accomodates both batch opertion as well as single vector operation, hence a try catch is used
        inpbool=np.asarray(inpbool)
        if(np.size(inpbool)>1):
            self.temp=torch.FloatTensor(np.asanyarray([1.0 if (i ==True) else 0.0 for i in inpbool ]))
            # Boolean inputs are converted into float tensors
            self.inp=torch.FloatTensor(torch.np.ones((2,np.size(inpbool))))
            self.inp[1,:]=self.temp
            self.out=self.forward()
            self.final_op=[True if (j >0.5) else False for i in self.out for j in i]    
        else:
            self.temp=torch.FloatTensor(np.asanyarray([1.0 if (i ==True) else 0.0 for i in inpbool ]))
            self.inp=torch.cat((torch.FloatTensor(torch.np.ones(1)),self.temp),0)        
            self.out=self.forward()
            # The float tensors output from forward propagation of NN is converted back into boolean
            self.final_op=[True if (i >0.5) else False for i in self.out]
        return self.final_op
    def forward(self):
        # This function calls the forward operation in the NeuralNetwork class to compute the forward propagation
        return NeuralNetwork.forward(self,self.inp)

class XOR(NeuralNetwork):
    def __init__(self):
        NeuralNetwork.__init__(self,2,2,1)
        # Get the layer and modify the values for the layer to suit the AND operation
        # The threshold for True or False is 0.5
        self.layer=NeuralNetwork.getlayer(self,0)
        self.layer[0,:]=-5
        self.layer[1,0]=4
        self.layer[2,0]=4
        self.layer[1,1]=6
        self.layer[2,1]=6
        self.layer=NeuralNetwork.getlayer(self,1)
        self.layer[0]=-1
        self.layer[1]=-2
        self.layer[2]=2.5
        return
    def __call__(self,inpbool):
        # Input is considered to be a list, hence converted into an array
        # This function accomodates both batch opertion as well as single vector operation, hence a try catch is used
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
            self.out=self.forward()
            self.final_op=[True if (j >0.5) else False for i in self.out for j in i]    
        except IndexError:
            self.temp=torch.FloatTensor(np.asanyarray([1.0 if (i ==True) else 0.0 for i in inpbool ]))
            self.inp=torch.cat((torch.FloatTensor(torch.np.ones(1)),self.temp),0)        
            self.out=self.forward()
            # The float tensors output from forward propagation of NN is converted back into boolean
            self.final_op=[True if (i >0.5) else False for i in self.out]
        return self.final_op
    def forward(self):
        # This function calls the forward operation in the NeuralNetwork class to compute the forward propagation
        return NeuralNetwork.forward(self,self.inp)


