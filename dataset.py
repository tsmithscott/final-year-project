import tensorflow as tf
from matplotlib import pyplot as plt

root_dir = 'C:/Programming/FinalYearProject/dataset512x512'

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

# Print the class names
class_names = train_ds.class_names
print(class_names)

# Visualize the dataset
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i].numpy()[0].astype(int)])
    plt.axis("off")
plt.show()