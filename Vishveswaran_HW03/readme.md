# Assumptions:
 1. In NeuralNetwork API, hidden layers passed as the parameters for build does not include the bias term for each layer.
 2. In NeuralNetwork API, weights for the bias term is considered as per the requirements provided in the instruction(this is included in the random distribution).
 3. In LogicGates API, the input can be array as well as list.


To test the API, a test script is generated and all the test cases are analysed. Inputs as well as results are printed for ease of reference. 
To execute the test script, open the test script in any python IDE and run it. Else use a terminal window and type "$ python testscript.py"
It will print the inputs and the outputs correspondingly. Both batch operation and single operation are tested.

# Results & Discussion:
1. Training with stochastic process and batch gradient are implemented as per instruction.
2. Learning rate eta is set for each gate internally to get optimum result with set no.of iterations
3. The final weights that will be used for testing and training accuracy after the last training iteration is printed out for reference
4. The weights vary with each run since the weights are initialized randomly.


# The Bonus part of Cross entropy was implemented too.
# To activate the cross entropy do the following:
	1. Go to the logic_gates.py file
	2. Go to the train() function of each gate
	3. Remove the comment over before the text 'CE' and the ')' before the comment symbol
	4. Execute the testscript.py file
	5. To get the MSE just dont pass 'CE' along with target into the backward() function in the train() for each gate
**The result displayed are due to back propagation with cross entropy.**
