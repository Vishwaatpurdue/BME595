# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 03:22:58 2017

@author: vishwa
"""

from conv import Conv2D
import torch
import numpy as np
import cv2
import timeit
import matplotlib.pyplot as plt
# Loading Images
path1="./img1.jpg"
image1=cv2.imread(path1)
C1=image1[:,:,0]
C2=image1[:,:,1]
C3=image1[:,:,2]
image=np.zeros((3,image1.shape[0],image1.shape[1]),dtype=float)
image[0,:,:]=C1
image[1,:,:]=C2
image[2,:,:]=C3
o_channel=3
print(image.shape)
# Part-A
#conv2D=Conv2D(3,o_channel,3,2,'known')
#no_ops,output_temp=conv2D.forward(image)
#output=output_temp.numpy()
#print (no_ops)
#for i in range(o_channel):
#    cv2.imwrite("Task3_img2_channel_"+str(i)+".jpg",output[i,:,:])

## Part- B
#L_channels= np.asarray([2**i for i in range(5)])
#L_time=np.zeros((5,1),dtype='float32')
#for i in range(len(L_channels)):
#    #L_channels[i]=2**i
#    o_channel=L_channels[i]
#    conv2D=Conv2D(3,o_channel,3,1,'rand')
#    start_time = timeit.default_timer()
#    no_ops,output_temp=conv2D.forward(image)
#    L_time[i]=timeit.default_timer() - start_time
#    print(timeit.default_timer() - start_time)
##    output=output_temp.numpy()
##    cv2.imwrite("Dummy"+str(i)+".jpg",output[i,:,:])
#print(L_time)
#fig = plt.figure()
#a=plt.subplot(111)
#a.plot(L_channels,L_time)
#plt.title('Channels Vs Time')
#fig.savefig('./Figure_PartB.png')
#plt.close(fig)

#  Part- C
kernel_size=np.asarray([3,5,7,9,11])
L_ops=np.zeros((5,1))
for i in range(len(kernel_size)):
    conv2D=Conv2D(3,2,kernel_size[i],1,'rand')
    no_ops,output_temp=conv2D.forward(image)
    L_ops[i]=no_ops
fig = plt.figure()
a=plt.subplot(111)
a.plot(kernel_size,L_ops)
plt.title('Kernel_Size Vs No.of.Operations')
fig.savefig('./Figure_PartC.png')
