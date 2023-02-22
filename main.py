import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils.preprocessing import normalize_data
from utils.visualisation import plot_model_performance, visualize_images, visualize_augmented_images

# Windows
root_dir = 'C:/Programming/FinalYearProject/dataset512x512'
# # Mac
# root_dir = '/Users/theo/VSCode/FinalYearProject/dataset512x512'


# Load the data and split it into training and validation sets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    root_dir,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(512, 512),
    batch_size=32,
    label_mode="binary"
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    root_dir,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(512, 512),
    batch_size=32,
    label_mode="binary"
)


# Print the class names
class_names = train_ds.class_names
print(class_names)


# Visualize the original images
visualize_images(train_ds, class_names)

# Visualize augmented images
visualize_augmented_images(train_ds)


# Normalise the data to the range [0, 1] and print the min and max values
# Not used in the model currently
normalize_data(train_ds)


# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Create the model
num_classes = len(class_names)
model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(512, 512, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])


# Compile the model with an optimization function
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Print a summary of the model
model.summary()


# Train the model
epochs=50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)


# Analyze the model
plot_model_performance(history, epochs)
