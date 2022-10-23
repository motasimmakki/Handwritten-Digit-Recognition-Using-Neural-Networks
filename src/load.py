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

# Loading the trained model.
model = tf.keras.models.load_model('handwritten.model')

# Evaluate the model over the test model.
loss, accuracy = model.evaluate(x_test, y_test)

# Printing the loss and accuracy of the trained model.
# Gives absolute value, it should be as low as possible.
print(loss)
# Gives value between 0 & 1, 1 means 100% accuracy.
print(accuracy)
