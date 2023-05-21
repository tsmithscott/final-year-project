import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50, EfficientNetB0

path = 'C:/Programming/FinalYearProject/dataset512x512'
# path = '/Users/theo/VSCode/FinalYearProject/dataset512x512'

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


# Calculate and print model performance metrics
def print_metrics(y_true_classes, y_pred_classes):
    tp = tf.keras.metrics.TruePositives()
    tn = tf.keras.metrics.TrueNegatives()
    fp = tf.keras.metrics.FalsePositives()
    fn = tf.keras.metrics.FalseNegatives()
    tp.update_state(y_true_classes, y_pred_classes)
    tn.update_state(y_true_classes, y_pred_classes)
    fp.update_state(y_true_classes, y_pred_classes)
    fn.update_state(y_true_classes, y_pred_classes)

    accuracy = (tp.result() + tn.result()) / (tp.result() + tn.result() + fp.result() + fn.result())
    precision = tp.result() / (tp.result() + fp.result())
    recall = tp.result() / (tp.result() + fn.result())
    f1_score = 2 * precision * recall / (precision + recall)

    print("Accuracy: " + str(round(accuracy.numpy(), 3)))
    print("Precision: " + str(round(precision.numpy(), 3)))
    print("Recall: " + str(round(recall.numpy(), 3)))
    print("F1 Score: " + str(round(f1_score.numpy(), 3)))


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
    
    # Training
    custom_model.fit(train_generator, epochs=50, validation_data=validation_generator)
    
    # Cross-validation
    scores = custom_model.evaluate(validation_generator, verbose=0)
    
    # Prediction
    y_pred = custom_model.predict(validation_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get the true classes and calculate the confusion matrix
    y_true_classes = validation_generator.classes
    conf_matrix = tf.math.confusion_matrix(labels=y_true_classes, predictions=y_pred_classes)
    
    # Print the metrics
    print_metrics(y_true_classes, y_pred_classes)
    print("Custom Keras CNN Accuracy: %.2f%%" % (scores[1]*100))
    tf.print(conf_matrix)


def train_evaluate_pretrained_model(model_name):
    print(f"Checking {model_name} model...")
    
    # Define the model
    if model_name == "vgg16":
        base_model = VGG16(include_top=False, weights="imagenet", input_shape=(128, 128, 3))
    elif model_name == "inceptionv3":
        base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=(128, 128, 3))
    elif model_name == "resnet50":
        base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(128, 128, 3))
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    model = keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(128, 128),
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(2, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    # Cross-validation
    scores = model.evaluate(validation_generator, verbose=0)
    
    # Prediction
    y_pred = model.predict(validation_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get the true classes and calculate the confusion matrix
    y_true_classes = validation_generator.classes
    conf_matrix = tf.math.confusion_matrix(labels=y_true_classes, predictions=y_pred_classes)
    
    # Print the metrics
    print_metrics(y_true_classes, y_pred_classes)
    print(f"{model_name.capitalize()} Accuracy: {scores[1] * 100:.2f}%")
    tf.print(conf_matrix)


# Compare the models
check_custom_model()
train_evaluate_pretrained_model("vgg16")
train_evaluate_pretrained_model("inceptionv3")
train_evaluate_pretrained_model("resnet50")