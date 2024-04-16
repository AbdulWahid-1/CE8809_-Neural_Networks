# This is the output or main file
# I'm importing  librries I need
# Some are also from the class in different files we created
# the game takes output points from this file to ppredict next location of rocket
from myNeuralNetwork import NN
from myNeurons import NNeuron
from myDepth import Depth
import numpy as np
import pandas as pd
class NeuralNetHolder:
    # definig the length of the neural netork
    def __init__(self, size=[2, 4, 2]):
        self.mse = 0
        self.myLayers = list()
        for x in range(1, len(size)):
            dpth = Depth(size[x], size[x - 1])
            self.myLayers.append(dpth)
    # define our feed forward function
    def feedForward(self, inputs):
        # creting a empty list as our output variable
        predicted = []
        predicted = inputs
        for x in self.myLayers:
                improved = []
                # AFter applying ou activation a eahc neuron the progam will update each neuron
                for nrn in x.depth:
                    improved.append(nrn.activation(predicted))
                predicted = improved
        return predicted
    # normalize the data we receive from the game as an input with min max technique
    def nrmlzd(self,dta):
        temp = dta
        normalized = []
        normalized = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
        return  normalized
    # the game still runs on denormalized data. we nmormalized it for our need
    # through this function we will convert the output of our custom neural network into game required one
    def denormalize(self, dta, x):
        # x = original data before normalization
        denormalized = []
        denormalized = float(((x*(np.max(dta) - np.min(dta)))/np.min(dta)))
        return  denormalized
    # this is like a main function responsible for all tasks
    # it takes a kind of string as an input
    # i'm splitting it and converting it into a form of list so I can work properly on it
    def predict(self, input_row):
        # converting inputs into list
        x1=input_row.split(",")[0]
        x1=float(x1)
        x2=input_row.split(",")[1]
        x2=float(x2)
        lst = []
        lst.append(x1)
        lst.append(x2)

        # creating another empty list and performing normalization
        lst1 = []
        lst1 = self.nrmlzd(lst)
        print(lst)

        # Providing our optimized weights to the program to increase accuracy
        NNeuron.wghts = [[0.3750401167496323, 0.3656777339949322, 0.6802410898304506, 1.4452219695217157],  
                        [0.5915497794400468, 0.19690749464835772, 0.2880688611805924, 0.0846367706382636]]
        
        output =  NeuralNetHolder.feedForward(self, lst1)

        return output