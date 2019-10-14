# Weight-Initialization-for-Neural-Networks
_Initializations define the way to set the initial random weights of Keras layers._

## Introduction

To find good initial weights for a neural network. 
Having good initial weights can place the neural network close to the optimal solution. This allows the neural network to come to the best solution quicker.

Building even a simple neural network can be confusing task and upon that tuning it to get a better result is extremely tedious. But, the first step that comes in consideration while building a neural network is initialization of parameters, if done correctly then optimization will be achieved in least time otherwise converging to a minima using gradient descent will be impossible.

## Basic notations

Consider a L layer neural network, which has L-1 hidden layers and 1 input and output layer each.
### Zero initialization :

In general practice biases are initialized with 0 and weights are initialized with random numbers, what if weights are initialized with 0 ?

In order to understand this lets consider we applied sigmoid activation function for the output layer.

## Testing Weights
### Dataset

To see how different weights perform, we'll test on the same dataset and neural network. Let's go over the dataset and neural network.

We'll be using the MNIST dataset to demonstrate the different initial weights. As a reminder, the MNIST dataset contains images of handwritten numbers, 0-9, with normalized input (0.0 - 1.0). Run the cells to download and load the MNIST dataset.


![Neural Network](https://github.com/Ratna04priya/Weight-Initialization-for-Neural-Networks/blob/master/neural_network.png)
