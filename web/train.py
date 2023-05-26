import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, Resizing
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_model():
    
    model = Sequential([
        Rescaling(1./255, input_shape=(512, 512, 3)),
        Resizing(128, 128),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(128, 3, padding='same', activation='relu'),
        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(2, activation='sigmoid')
    ])

    # Compile the model with the specified learning rate and dropout rate
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model


def train_model():
    root_dir = "../dataset512x512/"

    # Define your train and validation datasets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        root_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(512, 512),
        batch_size=32,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb"
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        root_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(512, 512),
        batch_size=32,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb"
    )
    model = get_model()
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=50)
    model.save('model.h5')


if __name__ == "__main__":
    train_model()