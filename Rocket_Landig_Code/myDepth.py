# this comes rigt next afte rthe neuron class
# we define the depth/layers of our neural network
import numpy as np
from myNeurons import NNeuron
# creating my layer
class Depth:
    def __init__(self, n_nrns, n_inputs):
        self.depth = []
        #defining n number of nurons user defined
        for x in range(0,n_nrns):
            self.depth.append(NNeuron(n_inputs))
        print(self.depth)