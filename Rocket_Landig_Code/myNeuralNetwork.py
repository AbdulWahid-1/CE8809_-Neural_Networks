# thsi file is basically the shape of my neural network
# after creating the neuron an defining the layer I'm putting everything all together
from myDepth import Depth
import math 
import numpy as np
# this is the class that contains all the functions like, feedForward, BackPropragaton, training, predict, validation, error, RMSEe.t,c
class NN:
    # first we are defining the main things as global in class,
    # 1- MSE that we will use to calculate RMSE
    # 2- our layers, or depty of neural  network
    def __init__(self, size):
        self.mse = 0
        self.myLayers = list()
        for x in range(1, len(size)):
            dpth = Depth(size[x], size[x - 1])
            self.myLayers.append(dpth)
    
    def feedForward(self, inputs):
        predicted = inputs
        for x in self.myLayers:
                improved = []
                # we are just impleting activation function on each  neuron we created in our neural network
                for nrn in x.depth:
                    improved.append(nrn.activation(predicted))
                predicted = improved
        return predicted


    # It calcuates and sets values of parameters
    def backPropagation(self, expected):
        for x in reversed(range(len(self.myLayers))):
            lyr = self.myLayers[x]
            # initializing our list of errors
            error = []
            # if the inputs are not equal to one less than the length of our layer
            # which means if we're not in first hidden layer
            if x != len(self.myLayers) - 1:
                for y in range(len(lyr.depth)):
                    temp_error = 0.0
                    # for every neuron the error will be equal to sum of previous errors and dot product of deta and weights associated with the neuron
                    for nrn in self.myLayers[x + 1].depth:
                        temp_error  =  temp_error + (nrn.dlta * nrn.wghts[y])
                    error.append(temp_error)
            else:
                # if not the case error equals the difference of expected value and sgmoid on the neuron
                for y in range(len(lyr.depth)):
                    nrn = lyr.depth[y]
                    error.append(expected[y] - nrn.sgmd)
            # updating the value of delta afte reach iteration ofthe first loop
            for z in range(len(lyr.depth)):
                nrn = lyr.depth[z]
                nrn.dlta = nrn.actDerivative() * error[z]

    # in this function we upadate the weights
    def weightsUpdt(self, inputs, lr):
        for x in range(len(self.myLayers)):
            #if we're not in first hidden layer
            if x != 0: 
                # initialize list of inputs
                inputs = []
                # appending segmoid for all neurons in the layer
                for nrn in self.myLayers[x-1].depth:
                    inputs.append(nrn.sgmd)
            # this nested for loop mainly focuses on updating the weighgs
            for nrn in self.myLayers[x].depth:
                # new weights are equal to prevoius weghts as this location plus products of learning rate, inputs at this position and value of delta
                for y in range(len(inputs)):
                    nrn.wghts[y] = nrn.wghts[y] + lr * inputs[y] * nrn.dlta
                # Update bias weight
                nrn.wghts[-1] = nrn.wghts[-1] + lr * nrn.dlta
                print(nrn.wghts)
    # I'm calculating MSE in order to calculate RMSE
    def MSE(self, dta, expected, inputs):
        for x in dta:
            # from the given data we take inputs and expected
                inputs = x[0]
                expected = x[1]
                # we give it to feed forward to get predictions
                output = self.feedForward(inputs)
                # then we apply MSE formula
                temp = np.square(np.subtract(expected , output)).mean()
                # we each time keep adding the total value until the loop breaks
                self.mse = self.mse + temp
        # the value of mse is  then divided by number of inputs to get original value
        self.mse = self.mse / len(dta)
        return self.mse 
    # the sqrt root of mse is rmse
    def RMSE(self):
        rmse = 0
        rmse = math.sqrt(self.mse)
        return rmse
    # we trained our model wich takes generally 3 inouts
    # 1- dataset we will need for the problem
    # 2- learning rate
    # 3- Total number of epochs ir iterations
    def train(self, dta, lr, epoch):
        for iteration in range(epoch):
            # we are also calculating the training error so we can check difference between validation and training test
            total_training_error = 0.0
            # for training we first call feed forward
            for x in dta:
                inputs = x[0]
                expected = x[1]
                outputs = self.feedForward(inputs)
                # then we calcualte temporary error
                for y in range(len(expected)):
                    temp = expected[y] - outputs[y] 
                    temp = temp **2
                    total_training_error = total_training_error + temp
                    total_training_error = total_training_error / len(expected)
                # we apply our previously created functions in order
                # 1- backPropagation
                self.backPropagation(expected)
                # 2- weights updation
                self.weightsUpdt(inputs, lr)
                # 3 - MSE
                # 4- RMSE
                self.MSE(dta, expected, inputs)
                trnng_rmse = self.RMSE()
                # printing out the results
            print('epoch = ', (iteration), 'Total trainig error = ', total_training_error, 'learning rate= ', lr )
        print(expected)
        print('MSE = ',self.mse,'RMSE = ', trnng_rmse)
        # this function gives the total error value in testing set
    def testing (self, dta):
        total_testing_error = 0.0
        for x in dta:
                inputs = x[0]
                expected = x[1]
                outputs = self.feedForward(inputs)
                for y in range(len(expected)):
                    temp = expected[y] - outputs[y] 
                    temp = temp **2
                    total_testing_error = total_testing_error + temp
                    total_testing_error = total_testing_error / len(expected)
        print('testing Errror: ' ,total_testing_error)
    # doing the validation on the vaidation dataset
    def validation(self, dta):
        total_validation_error = 0.0
        for x in dta:
                inputs = x[0]
                expected = x[1]
                outputs = self.feedForward(inputs)
                for y in range(len(expected)):
                    temp = expected[y] - outputs[y] 
                    temp = temp **2
                    total_validation_error = total_validation_error + temp
                    total_validation_error = total_validation_error / len(expected)
        print('Validation Errror: ' ,total_validation_error)
    
    # thsi is the predict function which calls fed forard function in pogram
    def predict(self, inputs):
        return self.feedForward(inputs)