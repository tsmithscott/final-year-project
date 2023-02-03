import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def load_data(img_size=(32, 32), test_size=0.2, train_size=0.8):
    # Load the data into memory
    defect_dir = 'defective'
    non_defect_dir = 'non_defective'

    # Load the images
    defect_files = [os.path.join(defect_dir, f) for f in os.listdir(defect_dir) if f.endswith('.jpeg')]
    non_defect_files = [os.path.join(non_defect_dir, f) for f in os.listdir(non_defect_dir) if f.endswith('.jpeg')]

    # Read the images
    defect_imgs = [cv2.imread(f) for f in defect_files]
    non_defect_imgs = [cv2.imread(f) for f in non_defect_files]

    # Preprocess the data
    defect_imgs = [cv2.resize(img, img_size) for img in defect_imgs]
    non_defect_imgs = [cv2.resize(img, img_size) for img in non_defect_imgs]

    # Convert the data to numpy arrays
    defect_imgs = np.array(defect_imgs) / 255.0
    non_defect_imgs = np.array(non_defect_imgs) / 255.0

    # Create the labels
    defect_labels = [1 for _ in range(len(defect_imgs))]
    non_defect_labels = [1 for _ in range(len(non_defect_imgs))]

    # Combine the data and labels
    imgs = np.concatenate((defect_imgs, non_defect_imgs))
    labels = np.concatenate((defect_labels, non_defect_labels))
    
    # Split the data into training and validation sets
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(imgs, labels, test_size=test_size, train_size=train_size)
    
    # Convert the labels to one-hot vectors
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=2)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=2)

    # Data augmentation
    data_gen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    data_gen.fit(train_imgs)

    return train_imgs, train_labels, test_imgs, test_labels


# Load the data
train_imgs, train_labels, val_imgs, val_labels = load_data()

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_imgs, train_labels, epochs=10, batch_size=32, validation_data=(val_imgs, val_labels))

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_imgs, val_labels)
print("Validation loss:", val_loss)
print("Validation accuracy:", val_acc)
