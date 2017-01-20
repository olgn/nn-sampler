# This script will contain the functions and algorithms to train the neural network on the datasets output by preprocessing.py

import numpy as np
import scipy as sp
from sklearn import neural_network

#WARM START IS UR FRIEND


# define the structure of the network
	#how many layers
	#if we have A layers, each A-1 has 2x more nodes
hiddenLayerSizes = (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2)

#define our weighting function
activationFunction  = 'tanh'

#create the network using sk-learn : use architecture defined above, 
# mlp = neural_network.MLPClassifier(hidden_layer_sizes=hiddenLayerSizes, activation=activationFunction, warm_start=True, verbose=True, max_iter=200)
mlp = neural_network.MLPRegressor(hidden_layer_sizes=hiddenLayerSizes, activation=activationFunction, warm_start=True, verbose=True, tol=0.000001)
