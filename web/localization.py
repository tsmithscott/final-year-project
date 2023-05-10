import os
import random

from PIL import Image
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  array_to_img, img_to_array)


if not os.path.exists('model.h5'):
    model = train_model()
model = load_trained_model()


def load_trained_model():
    return tf.keras.models.load_model('model.h5')

def plot_dataset_activation():
    # Count the number of defective images
    num_defective = sum([is_defective(img) for img in X])
    
    # Determine the number of rows based on the number of defective images
    num_rows = num_defective
    
    # Create the subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(8, 8))
    
    # Initialize a counter for the row index
    row_index = 0
    
    # Iterate over the images
    for i, img in enumerate(X):
        if is_defective(img):
            weights = model.layers[-1].get_weights()[0]
            class_weights = weights[:, 0]

            intermediate = tf.keras.Model(model.input, model.get_layer("block5_conv3").output)
            conv_output = intermediate.predict(img[np.newaxis,:,:,:])
            conv_output = np.squeeze(conv_output)

            h = int(img.shape[0]/conv_output.shape[0])
            w = int(img.shape[1]/conv_output.shape[1])

            activation_maps = sp.ndimage.zoom(conv_output, (h, w, 1), order=1)
            out = np.dot(activation_maps.reshape((img.shape[0]*img.shape[1], 512)), class_weights).reshape(
                img.shape[0],img.shape[1])

            axes[row_index][0].imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
            axes[row_index][1].imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
            axes[row_index][1].imshow(out, cmap='jet', alpha=0.35, extent=[0, img.shape[1], img.shape[0], 0])
            axes[row_index][0].set_title('Defective')
            axes[row_index][1].set_title('Activation Map')
            
            # Increment the row index
            row_index += 1

    plt.tight_layout()
    plt.show()

def is_defective(image_path):
    with Image.open(image_path) as img:
        image = np.array(img)
    
    # Normalize the image
    image = image / 255.0
    
    # Make a prediction on the image using the model
    pred = model.predict(image[np.newaxis,:,:,:])
    
    # Check if the predicted class is 0 (defective)
    if np.argmax(pred) == 0:
        return True
    else:
        return False


def save_activation_map(image_path):
    # Load the image file and convert it to a NumPy array
    with Image.open(image_path) as img:
        image = np.array(img)

    weights = model.layers[-1].get_weights()[0]
    class_weights = weights[:, 0]

    intermediate = tf.keras.Model(model.input, model.get_layer("block5_conv3").output)
    conv_output = intermediate.predict(image[np.newaxis,:,:,:])
    conv_output = np.squeeze(conv_output)

    h = int(image.shape[0]/conv_output.shape[0])
    w = int(image.shape[1]/conv_output.shape[1])

    activation_maps = sp.ndimage.zoom(conv_output, (h, w, 1), order=1)
    out = np.dot(activation_maps.reshape((image.shape[0]*image.shape[1], 512)), class_weights).reshape(
        image.shape[0],image.shape[1])

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(image)
    axs[1].imshow(out, cmap='jet', alpha=0.35)
    axs[1].set_title('Activation Map Overlay')
    axs[1].axis('off')

    # Save the figure to the tmp folder
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    fig.savefig(os.path.join('tmp', f'{file_name}-output.png'))
    plt.close(fig)
