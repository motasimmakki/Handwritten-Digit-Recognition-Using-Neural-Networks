import os
# For using computer vision, to load images, process images, etc.
import cv2
# For working with numpy arrays.
import numpy as np
# Used for visualization of the actual digits.
import matplotlib.pyplot as plt
# For machine learning part.
import tensorflow as tf

# Loading the trained model.
model = tf.keras.models.load_model('handwritten.model')

# Iterating over the digit images.
image_number = 1;
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        # We are not taking care of any color.
        img = cv2.imread(f"digits/digit{image_number}.png")[:, :, 0]
        # Converting img into list then into numpy array for computation.
        img = np.invert(np.array([img]))
        # Prediction the image.
        prediction = model.predict(img)
        
        # Printing the prediction.
        # argmax: gives the index of the field that has the highest number.
        print(f"This digit is probably: {np.argmax(prediction)}")
        # Showing the digit.
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        # Incrementing the image number to check for next image.
        image_number += 1
    