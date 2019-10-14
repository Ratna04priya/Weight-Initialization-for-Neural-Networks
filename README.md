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

For the neural network, we'll test on a 3 layer neural network with ReLU activations and an Adam optimizer. The lessons you learn apply to other neural networks, including different activations and optimizers.



## Initialize Weights

All Zeros or Ones

If we follow the principle of Occam's razor, we might think setting all the weights to 0 or 1 would be the best solution. This is not the case.

With every weight the same, all the neurons at each layer are producing the same output. This makes it hard to decide which weights to adjust.

Let's compare the loss with all ones and all zero weights using helper.compare_init_weights. This function will run two different initial weights on the neural network above for 2 epochs. It will plot the loss for the first 100 batches and print out stats after the 2 epochs (~860 batches). We plot the first 100 batches to better judge which weights performed better at the start.

## Uniform Distribution

A uniform distribution has the equal probability of picking any number from a set of numbers. We'll be picking from a continous distribution, so the chance of picking the same number is low. We'll use TensorFlow's tf.random_uniform function to pick random numbers from a uniform distribution.

## Baseline

Let's see how well the neural network trains using the default values for tf.random_uniform, where minval=0.0 and maxval=1.0.


## General rule for setting weights

The general rule for setting the weights in a neural network is to be close to zero without being too small. A good pracitce is to start your weights in the range of [−y,y]
where y=1/n−−√ (n

is the number of inputs to a given neuron).

Let's see if this holds true, let's first center our range over zero. This will give us the range [-1, 1).

## Too small

Let's compare [-0.1, 0.1), [-0.01, 0.01), and [-0.001, 0.001) to see how small is too small. We'll also set plot_n_batches=None to show all the batches in the plot.

## Normal Distribution

Unlike the uniform distribution, the normal distribution has a higher likelihood of picking number close to it's mean. To visualize it, let's plot values from TensorFlow's tf.random_normal function to a histogram.

tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

Outputs random values from a normal distribution.

 shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
 mean: A 0-D Tensor or Python value of type dtype. The mean of the normal distribution.
 stddev: A 0-D Tensor or Python value of type dtype. The standard deviation of the normal distribution.
 dtype: The type of the output.
 seed: A Python integer. Used to create a random seed for the distribution. See tf.set_random_seed for behavior.
 name: A name for the operation (optional).
 
 ## Truncated Normal Distribution

tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

Outputs random values from a truncated normal distribution.

The generated values follow a normal distribution with specified mean and standard deviation, except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.

shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
mean: A 0-D Tensor or Python value of type dtype. The mean of the truncated normal distribution.
stddev: A 0-D Tensor or Python value of type dtype. The standard deviation of the truncated normal distribution.
dtype: The type of the output.
seed: A Python integer. Used to create a random seed for the distribution. See tf.set_random_seed for behavior.
name: A name for the operation (optional).




_There's no difference between the two, but that's because the neural network we're using is too small. A larger neural network will pick more points on the normal distribution, increasing the likelihood it's choices are larger than 2 standard deviations._




