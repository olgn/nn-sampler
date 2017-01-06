# This script will contain the functions and algorithms to train the neural network on the datasets output by preprocessing.py

import numpy as np
import scipy as sp
from sklearn import neural_network

#WARM START IS UR FRIEND


# define the structure of the network
	#how many layers
	#if we have A layers, each A-1 has 2x more nodes
hiddenLayerSizes = (256, 128, 64, 32, 16, 8, 4, 2)

#define our weighting function
activationFunction  = 'logistic'

#create the network using sk-learn : use architecture defined above, 
network = neural_network.MLPClassifier(hidden_layer_sizes=hiddenLayerSizes, activation=activationFunction)