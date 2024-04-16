# so in this file i'm defining everything about my neuron
# this file is where everything actaully starts when it comes to program logistics
import numpy as np
from random import random
from operator import mul

# i'm creating the class NNeuron inorder to define our neuron in each layer

class NNeuron:
    def __init__(self, n_inputs):
        # declaring the value of delta in the begining
        self.dlta = 0.5   
        # initializing the weights as a list
        self.wghts = []
        print(self.wghts)
        # initializing the sigmoid just incase
        self.sgmd = 0.0
        # cheking of the  self.weights are impty, initialize the weights in random
        if bool(self.wghts) == False:
            for x in range(n_inputs):
                self.wghts.append(random())

    # this is my activation where basically i'm collecting value of z
    # and then apppying sigmoid function to it
    # after that this function returns the generated output a
    def activation(self, inputs):
        # using try and except to get the error if occurs
        try:
            x = sum(map(mul, inputs, self.wghts[:-1]))
            self.sgmd = 1 / (1+np.exp(-x))
            return self.sgmd
        except:
            print('There is error with the weights Mapping. confirm the weights')

    # taking derivative of the sigmoid function
    def actDerivative(self):
        try: 
            derv = 1 - self.sgmd ** 2
            return derv
        except:
            print('Sigmoid function is not returning any value')