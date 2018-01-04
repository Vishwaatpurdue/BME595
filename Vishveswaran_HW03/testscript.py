import numpy as np
import torch
from logic_gates import AND,OR,NOT,XOR

# Testing
a=AND()

c=OR()

b=XOR()

e=NOT()

i=[[True,False],[True,True]]
i1=[True,True]
i2=[False,False]
i3=[False,True]
print("The input sequence for testing is:")
print("1.",i)
print("2.",i1)
print("3.",i2)
print("4.",i3)
print("Training of AND gate")
a.train()
print("Testing of AND gate")
print(a.forward(i))
print(a.forward(i1))
print(a.forward(i2))
print(a.forward(i3))
print("Training of XOR gate")
b.train()
print("Testing of XOR gate")
print(b.forward(i))
print(b.forward(i1))
print(b.forward(i2))
print(b.forward(i3))
print("Training of OR gate")
c.train()
print("Testing of OR gate")
print(c.forward(i))
print(c.forward(i1))
print(c.forward(i2))
print(c.forward(i3))
print("Training of NOT gate")
e.train()
print("Testing of NOT gate")
print(e.forward(i3))