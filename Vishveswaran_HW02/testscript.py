#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 20:01:40 2017

@author: vishwa
"""
from logic_gates import AND,OR,NOT,XOR
from neural_network import NeuralNetwork
import torch
import numpy as np
# Testing the Neural Network
testNN=NeuralNetwork(1,1,1,1,2)
j=torch.FloatTensor(np.array([[1.0,2.0],[1.0,2.0]]))
print(testNN.getlayer(2))
print(testNN.forward(j))
testNN=NeuralNetwork(2,2,2,2,2)
j=torch.FloatTensor(np.array([1.0,2.0,3.0]))
print(testNN.forward(j))
###############################################################################
#ip_not=[True]
ip_not=[False]
#ip_not=[[True],[False]]
i=[[True,True],[True,True]]
#i=[[False,False],[True,True]]
#i=[[True,True],[False,False]]
#i=[[False,False],[False,False]]
#i=[True,True]
#i=[False,True]
# Initialization
And=AND()
Or=OR()
Not=NOT()
Xor=XOR()

# For NOT gate
ip_not=[True]
Not_op=Not(ip_not)
print(Not_op,",",ip_not)
ip_not=[False]
Not_op=Not(ip_not)
print(Not_op,",",ip_not)
print("\n Batch inputs")
# With multiple inputs (batch operation)
ip_not=[[True],[False]]
Not_op=Not(ip_not)
print(Not_op,",",ip_not)
# For other gates
print("\n Other gates (AND,OR,XOR) in the same order is displayed below")
print("It is in the format input , output")
print("Input is in row,row format")
i=[True,True]
And_op=And(i)
Or_op=Or(i)
Xor_op=Xor(i)
print(i,",",And_op)
print(i,",",Or_op)
print(i,",",Xor_op)
i=[True,True]
And_op=And(i)
Or_op=Or(i)
Xor_op=Xor(i)
print(i,",",And_op)
print(i,",",Or_op)
print(i,",",Xor_op)
i=[False,True]
And_op=And(i)
Or_op=Or(i)
Xor_op=Xor(i)
print(i,",",And_op)
print(i,",",Or_op)
print(i,",",Xor_op)
print("\n Batch inputs")
# batch inputs
i=[[True,True],[True,True]]
And_op=And(i)
Or_op=Or(i)
Xor_op=Xor(i)
print(i,",",And_op)
print(i,",",Or_op)
print(i,",",Xor_op)
i=[[False,False],[True,True]]
And_op=And(i)
Or_op=Or(i)
Xor_op=Xor(i)
print(i,",",And_op)
print(i,",",Or_op)
print(i,",",Xor_op)
i=[[True,True],[False,False]]
And_op=And(i)
Or_op=Or(i)
Xor_op=Xor(i)
print(i,",",And_op)
print(i,",",Or_op)
print(i,",",Xor_op)
i=[[False,False],[False,False]]
And_op=And(i)
Or_op=Or(i)
Xor_op=Xor(i)
print(i,",",And_op)
print(i,",",Or_op)
print(i,",",Xor_op)