# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 03:22:58 2017

@author: vishwa
"""

import torch
import numpy as np
class Conv2D:
    def __init__(self,*args):
        # Initializing all the necessary parameters to compute the convulusion
        self.i_channel=args[0]
        self.o_channel=args[1]
        self.kernel_size=args[2]
        self.mode=args[4]
        self.stride=args[3]
        self.k1=np.asarray([[-1,-1,-1],[ 0,0,0],[ 1,1,1]])
        self.k2=np.asarray([[-1,-1,-1],[ 0,0,0],[ 1,1,1]]).T
        self.k3=np.ones((3,3))
        self.k4=np.asarray([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[0,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1]])
        self.k5=np.asarray([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[0,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1]]).T
        if self.mode=='known':
            if self.o_channel==1:
                self.kernel=self.k1
            elif self.o_channel==2:
                self.kernel=np.asarray([[self.k4],[self.k5]])
            elif self.o_channel==3:
                self.kernel=np.asarray([[self.k1],[self.k2],[self.k3]])
        return
    def forward(self,image):
        # Computing the convulusion in accordance to parameters defined during initialization
        # The input image is provided in C X H X W
        no_ops=0
        try:
            if image.type()[:5]=='torch':
                img=image.numpy()
        except AttributeError:
            img=image
            
        img1=np.zeros((self.o_channel,(image.shape[1]-self.kernel_size)/self.stride+1,(image.shape[2]-self.kernel_size)/self.stride+1))
        if self.mode=='known':
            if self.o_channel==1:
                xloop=0
                for x in xrange(0,image.shape[1]-self.kernel_size+1,self.stride):                 
                    yloop=0
                    for y in xrange(0,image.shape[2]-self.kernel_size+1,self.stride):
                        temp1=np.sum((img[0,x:x+self.kernel_size,y:y+self.kernel_size]*self.kernel))
                        temp2=np.sum((img[1,x:x+self.kernel_size,y:y+self.kernel_size]*self.kernel))
                        temp3=np.sum((img[2,x:x+self.kernel_size,y:y+self.kernel_size]*self.kernel))
                        img1[0,xloop,yloop]=temp1+temp2+temp3
                        no_ops=no_ops+1
                        yloop=yloop+1                        
                    xloop=xloop+1   
                return no_ops, torch.from_numpy(img1)
            for i in xrange(self.o_channel):
                xloop=0
                for x in xrange(0,image.shape[1]-self.kernel_size+1,self.stride):
                    yloop=0
                    for y in range(0,image.shape[2]-self.kernel_size+1,self.stride):
                        temp1=np.sum((img[0,x:x+self.kernel_size,y:y+self.kernel_size]*self.kernel[i,:,:]))
                        temp2=np.sum((img[1,x:x+self.kernel_size,y:y+self.kernel_size]*self.kernel[i,:,:]))
                        temp3=np.sum((img[2,x:x+self.kernel_size,y:y+self.kernel_size]*self.kernel[i,:,:]))
                        img1[i,xloop,yloop]=temp1+temp2+temp3
                        no_ops=no_ops+1
                        yloop=yloop+1
                    xloop=xloop+1
        elif self.mode=='rand':
            self.kernel=np.random.rand(self.kernel_size,self.kernel_size)
            if self.o_channel==1:
                xloop=0                
                for x in xrange(0,image.shape[1]-self.kernel_size+1,self.stride):
                    yloop=0
                    for y in xrange(0,image.shape[2]-self.kernel_size+1,self.stride):
                        temp1=np.sum((img[0,x:x+self.kernel_size,y:y+self.kernel_size]*self.kernel))
                        temp2=np.sum((img[1,x:x+self.kernel_size,y:y+self.kernel_size]*self.kernel))
                        temp3=np.sum((img[2,x:x+self.kernel_size,y:y+self.kernel_size]*self.kernel))
                        img1[0,xloop,yloop]=temp1+temp2+temp3
                        no_ops=no_ops+1
                        yloop=yloop+1
                    xloop=xloop+1
                return no_ops, torch.from_numpy(img1)
            for i in range(self.o_channel):
                self.kernel=np.random.rand(self.kernel_size,self.kernel_size)
                xloop=0                                
                for x in xrange(0,image.shape[1]-self.kernel_size+1,self.stride):
                    yloop=0                
                    for y in xrange(0,image.shape[2]-self.kernel_size+1,self.stride):
                        temp1=np.sum((img[0,x:x+self.kernel_size,y:y+self.kernel_size]*self.kernel))
                        temp2=np.sum((img[1,x:x+self.kernel_size,y:y+self.kernel_size]*self.kernel))
                        temp3=np.sum((img[2,x:x+self.kernel_size,y:y+self.kernel_size]*self.kernel))
                        img1[i,xloop,yloop]=temp1+temp2+temp3
                        no_ops=no_ops+1
                        yloop=yloop+1
                    xloop=xloop+1            
        return no_ops, torch.from_numpy(img1)
