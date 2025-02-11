import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import tensorflow as tf
from PIL import Image

from train import train_model
from tensorflow.keras.models import load_model

matplotlib.use('Agg')

if not os.path.exists('transfer-trained-vgg16.h5'):
    print("Model not found!\nTraining model...")
    model = train_model()
print("Loading trained model...")
model = load_model('transfer-trained-vgg16.h5')


def is_defective(image_path):
    with Image.open(image_path) as img:
        img = img.resize((512, 512))
        image = np.array(img)
    
    # Normalize the image
    image = image / 255.0
    
    # Make a prediction on the image using the model
    pred = model.predict(image[np.newaxis,:,:,:])
    confidence_percentage = np.max(pred) * 100
    
    # Check if the predicted class is 0 (defective)
    if np.argmax(pred) == 0:
        return True, confidence_percentage
    else:
        return False, confidence_percentage


weights = model.layers[-1].get_weights()[0]
class_weights = weights[:, 0]
intermediate = tf.keras.Model(model.input, model.get_layer("block5_conv3").output)

def save_activation_map(image_path):
    # Load the image file and convert it to a NumPy array
    with Image.open(image_path) as img:
        img = img.resize((512, 512))
        image = np.array(img)
    
    conv_output = intermediate.predict(image[np.newaxis,:,:,:])
    conv_output = np.squeeze(conv_output)

    h = int(image.shape[0]/conv_output.shape[0])
    w = int(image.shape[1]/conv_output.shape[1])

    activation_maps = sp.ndimage.zoom(conv_output, (h, w, 1), order=1)
    out = np.dot(activation_maps.reshape((image.shape[0]*image.shape[1], 512)), class_weights).reshape(
        image.shape[0],image.shape[1])

    fig, axs = plt.subplots(figsize=(6, 6))
    axs.imshow(image)
    axs.imshow(out, cmap='jet', alpha=0.35)
    axs.axis('off')
    plt.tight_layout()

    # Save the figure to the tmp folder
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    fig.savefig(os.path.join('tmp', f'{file_name}-output.png'))
    plt.close(fig)
