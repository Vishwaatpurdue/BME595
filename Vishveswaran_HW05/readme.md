# Assumptions
1. CNN is designed for both batch operation with Cross Entropy as loss function while SDG with single operation uses MSE as loss function.
2. Learning rate used is 0.2
3. Learning rate used for part B is 0.05. Also Lenet 5 model is modified to suit the training classes.
4. Regularization is not implemented for both cases.
# Testing
**PART A**
Run the test script named **testscript_partA.py** to get the required results.
To modify the no.of epochs, do the following:
1. Open **Img2Num.py** file
2. Change the **epochs** variable to no.of epochs required
3. Re-run the test script

**Part B**
To test the requirements, run the test script name **testscript_partB.py** 
1. No. of epochs used for training is 20. hence the training efficiency is around 49% but it mainly depends on the initialization of the weights.

# Results and Discussion
**PART A**
The convergence around 30 epochs with total error around 8. For both NN and CNN.
The time consumption for training linearly increases with no.of epochs.
1. Figure 1 represents CNN Vs NN w.r.to Total Error
2. Figure 2 represents CNN Vs NN w.r.to Training time
3. Figure 3 represents CNN Vs NN w.r.to Inference time

1. From charts the training time and inference time for the NN is **faster** than CNN
2. But the Total Error is significantly less using CNN than NN.
3. Training for 30 epochs the training error falls to a total of 7/50000

**Part B**
1. The main issue is local minimum obtained when the epoch batch loss is around 4.6. It takes atleat 10 epoch to get out of that minima.
2. Due to small window size the caption detected for the live camera is printed as well as display as frame header. But Frame header could not be seen.
3. Training error is printed for after completion of every epoch for the last batch, for ease of understanding and verification
