import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    

def get_model(SHAPE: tuple):
    set_seed(33)
    
    vgg = VGG16(weights='imagenet', include_top=False, input_shape = SHAPE)

    for layer in vgg.layers[:-8]:
        layer.trainable = False

    x = vgg.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(vgg.input, x)
    model.compile(loss = "categorical_crossentropy",
                  optimizer = Adam(learning_rate=0.0001), 
                  metrics=["accuracy"])
    
    return model


def train_model():
    # Define variables
    SHUFFLE = True
    SHAPE = (512, 512, 3)
    batch_size = 8

    # Load and preprocess the datasets
    root_dir = 'C:/Users/theos/PycharmProjects/FinalYearProject/dataset512x512'
    train_datagen = ImageDataGenerator(
            validation_split=0.2, rescale=1./255
    )
    validation_datagen = ImageDataGenerator(
            validation_split=0.2, rescale=1./255
    )
    train_generator = train_datagen.flow_from_directory(
                root_dir,
                target_size = (SHAPE[0], SHAPE[1]),
                batch_size = batch_size,
                class_mode = 'categorical',
                shuffle = SHUFFLE,
                subset = 'training',
                seed = 33
    )
    validation_generator = validation_datagen.flow_from_directory(
                root_dir,
                target_size = (SHAPE[0], SHAPE[1]),
                batch_size = batch_size,
                class_mode = 'categorical',
                shuffle = SHUFFLE,
                subset = 'validation',
                seed = 33
    )
    model = get_model(SHAPE)
    model.fit(train_generator,
              validation_data=validation_generator,
              epochs=50)
    model.save('transfer-trained-vgg16.h5')


if __name__ == "__main__":
    train_model()