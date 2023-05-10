import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  array_to_img, img_to_array)

### DEFINE SOME PARAMETERS ###
base_path = "../dataset512x512/"
SHAPE = (512,512,3)
batch_size = 8

### INITIALIZE GENERATORS ###
train_datagen = ImageDataGenerator(
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


train_model()