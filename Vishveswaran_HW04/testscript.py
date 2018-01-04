import numpy as np
import torch
from MyImg2Num import MyImg2Num
import timeit
from torchvision import datasets, transforms
from NnImg2Num import NnImg2Num
from torch.autograd import Variable
## Testing
a=MyImg2Num()
st_time=timeit.default_timer()
a.train()
time_used=timeit.default_timer()-st_time
print("The timetaken to train the given model is "+str(time_used))
# To test the forward of the MyImg2Num is given below
test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])))
testdata=torch.FloatTensor(torch.zeros(len(test_loader),785))
test_label=torch.FloatTensor(np.zeros(len(test_loader)))
test_pred=torch.FloatTensor(np.zeros(len(test_loader)))
# For testing the forward function of the MyImg2Num class
for idx,(data,target) in enumerate(test_loader):
        testdata[idx,1:]=data.view(-1)
        testdata[idx,0]=1.0
        test_label[idx]=target[0]
        test_pred[idx]=a.forward(data)
#
############################
## Testing NN in pytorch
NnImg=NnImg2Num()
start_time=timeit.default_timer()
NnImg.train()
time_used=timeit.default_timer()-start_time
print("The time taken for training is "+str(time_used))
train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),batch_size=200)
f_err=0
# Inference
start_time=timeit.default_timer()
for idx,(data,target) in enumerate(train_loader): 
    data=Variable(data,volatile=True)
    target=Variable(target)
    op=NnImg.forward(data)
    tgt1=(target.data).numpy()
    op1=(op.data).numpy()
    comp=np.asarray([1 if tgt1[i]!=op1[i] else 0 for i in range(200)])
    f_err=f_err+np.sum(comp)
time_used=timeit.default_timer()-start_time
print("The time taken for inference is "+str(time_used))
