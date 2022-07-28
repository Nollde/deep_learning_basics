"""
Exercise 1: Introduction to the TensorFlow API
"""

import numpy as np
import tensorflow as tf


# Constants, sequences and random values, similar to numpy
print("\nConstants\n=====================")

t1 = tf.ones([3, 2])
t2 = tf.zeros([5])
t3 = tf.random.uniform([1, 3])
t4 = tf.linspace(1.0, 7.0, 4)  # note the first 2 arguments are floats
t5 = tf.convert_to_tensor(np.linspace(1, 7, 4))

# all tensorflow expressions are evaluated in a "Session"
print("tf.ones([3, 2]) = %s" % t1)
print("tf.zeros([5]) = %s" % t2)
print("tf.random_uniform([1, 3]) = %s" % t3)
print("tf.linspace(1.0, 7.0, 4) = %s" % t4)
print("tf.convert_to_tensor( np.linspace(1, 7, 4) ) = %s" % t5)


# Variables (used extensively in NNs)
print("\nVariables\n=====================")

w = tf.Variable(tf.zeros([3, 2]))

# variables need to be initialized first
print("w = %s" % w)

# assign new values
w.assign(tf.ones([3, 2]))

# retrieve values
print("w = %s" % w)


# models are build as a class holding the Symbolic variables
# feeding is done using the high performance tf.data API (used for input + output in DNNs)
print("\nSymbolic variables\n=====================")

# create TF datasets
# datasets allow you to manage your data
dataset = tf.data.Dataset.from_tensor_slices(([8., 3., 0., 8., 2., 1.], [16., 6., 0., 16., 8., 2.]))  # input, output


# Models are defined using a class holding the adaptive variables
# usually a model holds you Neural Network
# init: Define the model. initialization of variables
# call: Which operations are performed in the model

class Model(object):
  def __init__(self):
    self.W = tf.Variable(.1)

  def __call__(self, x):
    return self.W * x


model = Model()  # build the model

for x, y in dataset:  # loop over dataset

    print("x", x, "y", y)

    with tf.GradientTape() as tape:

        current_loss = (y - model(x))**2  # calculate loss (FORWARD PASS)
        dW = tape.gradient(current_loss, model.W)  # calculate gradient (BACKWARD PASS)
        model.W.assign_sub(dW)  # update model parameters assign_sub -> x-=y (gradient descent)


# Mathematical operations and functions
print("\nMathematical functions\n=====================")

x = tf.linspace(0., 4., 5)

print("x =", x)
print("(x+1)**2 - 2) =", (x + 1.)**2 - 2.)
print("sin(x)", tf.sin(x))
print("sum(x)", tf.reduce_sum(x))


# TensorFlow is vectorized: mathematical operations work on scalars and (element-wise) on tensors of any shape
# For example, these expressions are allowed:
tf.sin(3.)
tf.sin([1., 2., 3.])
tf.sin(tf.linspace(0., 10., 20))
tf.sin(np.linspace(0, 10, 20))  # equivalent
tf.sin(tf.ones(shape=(2, 3, 4)))  # 2x3x4 tensor


# Operators (+, -, /, *) are available
a = tf.zeros(shape=(2, 3))
b = tf.ones(shape=(2, 3))
c = tf.ones(shape=(3, 2))

a + b  # same as tf.add(a, b)
a - b  # same as tf.subtract(a, b)
a * b  # same as tf.mul(a, b)
a / b  # same as tf.division(a, b)
# a + c  # doesn't work; tensors need to be of same shape
