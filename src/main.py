import os
# For using computer vision, to load images, process images, etc.
import cv2
# For working with numpy arrays.
import numpy as np
# Used for visualization of the actual digits.
import matplotlib.pyplot as plt
# For machine learning part.
import tensorflow as tf

# Taking dataset directly from tensorflow, we don't need to download it.
mnist = tf.keras.datasets.mnist
# (Pixel data, Classification)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing: every value will be between 0 & 1.
x_train = tf._keras.utils.normalize(x_train, axis = 1)
x_test = tf._keras.utils.normalize(x_test, axis = 1)

# Now we can start working with the Neural network.

# Basic Model OR Basic Sequential Model.
model = tf.keras.models.Sequential()
# Adding layer to model
model.add(tf.keras.layers.Flatten(input_shape = (28, 28))) # Flatten the 28*28 grid.
# Adding dense layer: where each neuron is connected to each other neuron.
model.add(tf.keras.layers.Dense(128, activation = 'relu')) # relu: Rectify Linear Unit. | 0 if -ve, then move up linearly.
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# Output layer, 10 units is to represent individual digits, 0 -> 9.
# softmax: Each neuron interpret some value & signals how likely the image is digit.
# It gives the probability of each digit to be the right answer.
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

# Compiling the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])    

# Fit the model OR Train the model.
# epochs: How many iteration/times the model is going to see all over again.
model.fit(x_train, y_train, epochs = 3)

# Save the trained model.
model.save('handwritten.model')
