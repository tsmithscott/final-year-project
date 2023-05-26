import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define variables
SHAPE = (128, 128, 3)
batch_size = 8
# Load and preprocess the datasets
root_dir = 'C:/Programming/FinalYearProject/dataset512x512'
train_datagen = ImageDataGenerator(
        validation_split=0.2, rescale=1./255
)
test_datagen = ImageDataGenerator(
        validation_split=0.2, rescale=1./255
)
train_generator = train_datagen.flow_from_directory(
            root_dir,
            target_size = (SHAPE[0], SHAPE[1]),
            batch_size = batch_size,
            class_mode = 'categorical',
            shuffle = False,
            subset = 'training',
            seed = 33
)
validation_generator = test_datagen.flow_from_directory(
            root_dir,
            target_size = (SHAPE[0], SHAPE[1]),
            batch_size = batch_size,
            class_mode = 'categorical',
            shuffle = False,
            subset = 'validation',
            seed = 33
)


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_custom_model():
    model = Sequential([
        Conv2D(32, 3, padding='same', activation='relu', input_shape=(128, 128, 3)),
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
    model.compile(optimizer=Adam(learning_rate=0.0001),
                        loss="categorical_crossentropy",
                        metrics=['accuracy'])
    
    return model


def get_model(name: str, SHAPE: tuple):
    set_seed(33)
    
    if name == "custom":
        return get_custom_model()
    
    if name == "vgg16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=SHAPE)
        base_model.summary()
    elif name == "inceptionv3":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=SHAPE)
        base_model.summary()
    elif name == "resnet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=SHAPE)
        base_model.summary()
    else:
        raise Exception("Invalid model name")
        
    for layer in base_model.layers[:-8]:
            layer.trainable = False    
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(2, activation="softmax")(x)
    
    model = tf.keras.Model(base_model.input, x)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss="categorical_crossentropy",
                metrics=['accuracy'])
    return model


def train_model(name: str, model):
    
    model.fit(train_generator,
              validation_data=validation_generator,
              steps_per_epoch=train_generator.samples/train_generator.batch_size,
              epochs=10)
    model.save(f'trained-{name}.h5')
    print(f"Saved model: trained-{name}.h5")
    
    
# Define a function to calculate TP, TN, FP, FN rates
def calculate_rates(y_true_labels, y_pred_labels):
    # Calculate the confusion matrix metrics
    tn = tf.keras.metrics.TrueNegatives()
    tn.update_state(y_true_labels, y_pred_labels)
    fp = tf.keras.metrics.FalsePositives()
    fp.update_state(y_true_labels, y_pred_labels)
    fn = tf.keras.metrics.FalseNegatives()
    fn.update_state(y_true_labels, y_pred_labels)
    tp = tf.keras.metrics.TruePositives()
    tp.update_state(y_true_labels, y_pred_labels)
    # Get the TP, TN, FP, FN rates
    tp_rate = tp.result().numpy()
    tn_rate = tn.result().numpy()
    fp_rate = fp.result().numpy()
    fn_rate = fn.result().numpy()

    return tp_rate, tn_rate, fp_rate, fn_rate


def calculate_metrics(y_true, y_pred, name: str):
    tp, tn, fp, fn = calculate_rates(y_true, y_pred)

    # Calculate the accuracy, precision, recall, and F1-score
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    print("\n===============================================")
    print("{}".format(name))
    # Print the confusion matrix
    print("Confusion Matrix:")
    print("           Predicted Negative   Predicted Positive")
    print("Actual Negative       {}                 {}".format(tn, fp))
    print("Actual Positive       {}                 {}".format(fn, tp))
    # Print the performance metrics
    print("Accuracy: {:.3f}".format(accuracy))
    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("F1-score: {:.3f}".format(f1_score))
    print("===============================================\n")


def evaluate_model(model, y_true, name: str):
    y_pred = model.predict(validation_generator, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)
    calculate_metrics(y_true, y_pred, name)


def train_models():
    ### TRAIN MODELS ###
    custom_model = get_custom_model()
    vgg16 = get_model("vgg16", SHAPE)
    inceptionv3 = get_model("inceptionv3", SHAPE)
    resnet50 = get_model("resnet50", SHAPE)
    trained_custom_model = train_model("custom", custom_model)
    trained_vgg16 = train_model("vgg16", vgg16)
    trained_inceptionv3 = train_model("inceptionv3", inceptionv3)
    trained_resnet50 = train_model("resnet50", resnet50)


def load_trained_models():
    ### LOAD TRAINED MODELS ###
    custom = tf.keras.models.load_model('trained-custom.h5')
    vgg16 = tf.keras.models.load_model('trained-vgg16.h5')
    inceptionv3 = tf.keras.models.load_model('trained-inceptionv3.h5')
    resnet50 = tf.keras.models.load_model('trained-resnet50.h5')
    
    return custom, vgg16, inceptionv3, resnet50


def evaluate_models(custom, vgg16, inceptionv3, resnet50):
    ### EVALUATE MODELS - DISABLE SHUFFLE ON DATASETS ### 
    y_true = validation_generator.classes
    evaluate_model(custom, y_true,"Custom Model")
    evaluate_model(vgg16, y_true,"VGG16 (Transfer Learning)")
    evaluate_model(inceptionv3, y_true,"InceptionV3 (Transfer Learning)")
    evaluate_model(resnet50, y_true,"ResNet50 (Transfer Learning)")


# train_models()
custom, vgg16, inceptionv3, resnet50 = load_trained_models()
evaluate_models(custom, vgg16, inceptionv3, resnet50)
