import tensorflow as tf

from utils.visualisation import (plot_model_performance,
                                 visualize_augmented_images, visualize_images)

# Windows
root_dir = 'C:/Programming/FinalYearProject/dataset512x512'
# # Mac
# root_dir = '/Users/theo/VSCode/FinalYearProject/dataset512x512'

# Load the data and split it into training and validation sets
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

# Print the class names
class_names = train_ds.class_names
print(class_names)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(512, 512, 3)),
    tf.keras.layers.experimental.preprocessing.Resizing(128,128),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])


# Compile the model with an optimization function
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
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

model.save('model.h5')

# Analyze the model's performance
plot_model_performance(history, epochs)
