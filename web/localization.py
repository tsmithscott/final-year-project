import os
import random

from PIL import Image
import cv2
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


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def get_model():
    set_seed(33)
    
    vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape = SHAPE)

    for layer in vgg.layers[:-8]:
        layer.trainable = False

    x = vgg.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(vgg.input, x)
    model.compile(loss = "categorical_crossentropy", 
                  optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9), metrics=["accuracy"])
    
    return model


def train_model():
    model = get_model()
    model.fit(train_generator, steps_per_epoch=train_generator.samples/train_generator.batch_size, epochs=50)
    model.save('model.h5')


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
    fig.savefig(os.path.join('tmp', f'{file_name}-overlayed.png'))
    plt.close(fig)
    

if not os.path.exists('model.h5'):
    model = train_model()
model = load_trained_model()



if __name__ == "__main__":
    ### DEFINE SOME PARAMETERS ###
    base_path = "../dataset512x512/"
    SHAPE = (512,512,3)
    batch_size = 8
    
    ### INITIALIZE GENERATORS ###
    train_datagen = train_datagen = ImageDataGenerator(
            validation_split=0.3, rescale=1/255
    )
    test_datagen = ImageDataGenerator(
            validation_split=0.3, rescale=1/255
    )
    
    ### FLOW GENERATORS ###
    train_generator = train_datagen.flow_from_directory(
                base_path,
                target_size = (SHAPE[0], SHAPE[1]),
                batch_size = batch_size,
                class_mode = 'categorical',
                shuffle = True,
                subset = 'training',
                seed = 33
    )
    test_generator = test_datagen.flow_from_directory(
                base_path,
                target_size = (SHAPE[0], SHAPE[1]),
                batch_size = batch_size,
                class_mode = 'categorical',
                shuffle = True,
                subset = 'validation',
                seed = 33
    )


    ### RETRIVE TEST LABEL FROM GENERATOR ###
    test_num = test_generator.samples
    label_test = []
    for i in range((test_num // test_generator.batch_size)+1):
        X,y = test_generator.next()
        label_test.append(y)
    label_test = np.argmax(np.vstack(label_test), axis=1)
    label_test.shape

    ### PERFORMANCE ON TEST DATA ###
    print(classification_report(label_test, np.argmax(model.predict(test_generator),axis=1)))
    # plot_dataset_activation()
    save_activation_map("C:\\Programming\\FinalYearProject\\dataset512x512\\defective\\cast_def_0_255.jpeg")