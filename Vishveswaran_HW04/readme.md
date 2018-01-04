# Assumptions:
 1. In NeuralNetwork API, hidden layers passed as the parameters for build does not include the bias term for each layer.
 2. Learning rate eta is set as 0.5 and batch size for batch operation is set as 200.
 3. Self implemntation has the **MSE loss** while pytorch version is implemented with **CE loss** since the loss function is not specified exclusively.
To test the API, a test script is generated and all the test cases are analysed.
To execute the test script, open the test script in any python IDE and run it. Else use a terminal window and type "$ python testscript.py"
Both batch operation and single operation are tested.


# Testing
The current mode for self implementation is in SDG mode without batch operation.
To perform batch operation do the following in the **MyImg2Num.py** file :
1. UnComment the line no 41
2. Uncomment line no from 56 to 59
3. Comment line no from 61 to 63
**The pytorch implementation is tested with relu layer and without relu layer**
The results with relu layer are better than the ones without relu. The total error without relu layer were around 5068~5814 from epoch 5 to 1 respectively, Hence that implementation is not shown.
# Results & Discussion:
The chart for representing the Epochs Vs Error and Epochs Vs Speed(time taken) is computed for 5 epochs for both self implementation and pytorch implementation of Neural network.
1. Training with stochastic process and batch gradient are implemented as per instruction.
2. Test script for the forward function for the self implemented Neural Network is also shown in the testscript
3. From the charts we can say that the Neural Network implemented in pytorch is much faster compared to the self implemented one.
4. From the charts we can say that the time taken is **directly proportional (linear)** to the no.of epochs
**Note:**
1. Figure 1 represents Epochs Vs Error graph of self implementation without batch operation.
2. Figure 2 represents Epochs Vs Time taken graph of self implementation without batch operation.
3. Figure 3 represents Epochs Vs Error graph of pytorch implementation of NN.
4. Figure 4 represents Epochs Vs Time taken graph of pytorch implementation of NN.
5. The plot for batch implementation by self is tested for few cases as it was time consuming sufficient data was not collected to use for graph display.
