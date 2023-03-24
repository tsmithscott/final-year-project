import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = 'C:/Programming/FinalYearProject/dataset512x512'

# Define the parameters for the ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

# Define the training dataset
train_generator = datagen.flow_from_directory(
    path,
    target_size=(512, 512),
    batch_size=32,
    class_mode='binary',
    subset='training')

# Define the validation dataset
validation_generator = datagen.flow_from_directory(
    path,
    target_size=(512, 512),
    batch_size=32,
    class_mode='binary',
    subset='validation')

# Define the training data and labels for sklearn models
X_train, y_train = [], []
for i in range(len(train_generator)):
    batch = train_generator[i]
    X_train.extend(batch[0])
    y_train.extend(batch[1])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Define the validation data and labels for sklearn models
X_val, y_val = [], []
for i in range(len(validation_generator)):
    batch = validation_generator[i]
    X_val.extend(batch[0])
    y_val.extend(batch[1])
X_val = np.array(X_val)
y_val = np.array(y_val)

# Reshape the data for sklearn models
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)


# Train and evaluate custom keras model
def check_custom_model():
    print("Checking custom model...")
    # Define the model
    custom_model = keras.Sequential([
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
    custom_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
    # Train the model
    custom_model.fit(train_generator, epochs=50, validation_data=validation_generator)
    # Cross-validate the model
    scores = custom_model.evaluate(validation_generator, verbose=0)
    print("Custom Keras CNN Accuracy: %.2f%%" % (scores[1]*100))


# Train and evaluate Random Forest model
def check_random_forest():
    # Define the model
    random_forest = RandomForestClassifier()
    # Train the model
    random_forest.fit(X_train, y_train)
    # Cross-validate the model
    scores = random_forest.score(X_val, y_val)
    print("Random Forest Accuracy: %.2f%%" % (scores*100))
    

# Train and evaluate Naive Bayes model
def check_naive_bayes():
    # Define the model
    naive_bayes = GaussianNB()
    # Train the model
    naive_bayes.fit(X_train, y_train)
    # Cross-validate the model
    scores = naive_bayes.score(X_val, y_val)
    print("Naive Bayes Accuracy: %.2f%%" % (scores*100))
    

# Train and evaluate SVM model
def check_svm():
    # Define the model
    svm = SVC()
    # Train the model
    svm.fit(X_train, y_train)
    # Cross-validate the model
    scores = svm.score(X_val, y_val)
    print("Support Vector Machine (SVM) Accuracy: %.2f%%" % (scores*100))


# Compare the models
check_custom_model()
check_random_forest()
check_naive_bayes()
check_svm()