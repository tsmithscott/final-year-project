import numpy as np
import tensorflow as tf


# Normalise the data to the range [0, 1]
def normalize_data(train_ds):
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x, training=True), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image))
    return normalized_ds
