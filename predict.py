import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model.h5')


# Define the function to create the annotations
def create_annotation(image, bboxes):
    # Draw bounding boxes around the detected objects
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def get_bounding_boxes(predictions, threshold=0.5):
    # Threshold the predictions to get binary values
    binary_predictions = np.where(predictions > threshold, 1, 0)
    # Find the coordinates of the bounding boxes
    bboxes = []
    for prediction in binary_predictions:
        y, x = np.where(prediction)
        if len(y) > 0 and len(x) > 0:
            ymin, ymax = min(y), max(y)
            xmin, xmax = min(x), max(x)
            bboxes.append((xmin, ymin, xmax, ymax))
        else:
            bboxes.append((0, 0, 0, 0))
    return bboxes


# Define the root directory and the input and output directories
root_dir = 'C:/Programming/FinalYearProject/dataset512x512'
input_dir = os.path.join(root_dir, 'defective')
output_dir = os.path.join(root_dir, 'output')

# Load the images from the input directory and create a dataset
image_paths = tf.io.gfile.glob(input_dir + '/*.jpeg')
print(image_paths)
print(type(image_paths[0]))
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(lambda x: tf.io.read_file(x))
dataset = dataset.map(lambda x: tf.image.decode_jpeg(x, channels=3))
dataset = dataset.batch(32)

for images, labels in dataset.take:
    predictions = model.predict(images)
    bboxes = get_bounding_boxes(predictions)
    annotated_images = tf.data.Dataset.from_tensor_slices((images, bboxes)).map(create_annotation)
    # Visualize the annotated images
    for annotated_image in annotated_images.take(1):
        plt.imshow(annotated_image[0].numpy())
        plt.show()

# Get the first image and its corresponding label from the dataset
first_image, first_label = next(iter(dataset))

# Make predictions for the first image
predictions = model.predict(tf.expand_dims(first_image, axis=0))
bboxes = get_bounding_boxes(predictions)

# Create an annotated image for the first image
annotated_image = create_annotation(first_image, bboxes)

# Visualize the annotated image
plt.imshow(annotated_image.numpy())
plt.show()