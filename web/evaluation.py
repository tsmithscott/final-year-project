import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import load_trained_model


if __name__ == "__main__":
    ### DEFINE SOME PARAMETERS ###
    model = load_trained_model()
    base_path = "../dataset512x512/"
    SHAPE = (512,512,3)
    batch_size = 8
    
    ### INITIALIZE GENERATORS ###
    train_datagen = train_datagen = ImageDataGenerator(
            validation_split=0.3, rescale=1/255
    )
    test_datagen = ImageDataGenerator(
            validation_split=0.3, rescale=1/255
    )
    
    ### FLOW GENERATORS ###
    train_generator = train_datagen.flow_from_directory(
                base_path,
                target_size = (SHAPE[0], SHAPE[1]),
                batch_size = batch_size,
                class_mode = 'categorical',
                shuffle = True,
                subset = 'training',
                seed = 33
    )
    test_generator = test_datagen.flow_from_directory(
                base_path,
                target_size = (SHAPE[0], SHAPE[1]),
                batch_size = batch_size,
                class_mode = 'categorical',
                shuffle = True,
                subset = 'validation',
                seed = 33
    )


    ### RETRIVE TEST LABEL FROM GENERATOR ###
    test_num = test_generator.samples
    label_test = []
    for i in range((test_num // test_generator.batch_size)+1):
        X,y = test_generator.next()
        label_test.append(y)
    label_test = np.argmax(np.vstack(label_test), axis=1)
    label_test.shape

    ### PERFORMANCE ON TEST DATA ###
    print(classification_report(label_test, np.argmax(model.predict(test_generator),axis=1)))
    # plot_dataset_activation()
    # save_activation_map("C:\\Programming\\FinalYearProject\\dataset512x512\\defective\\cast_def_0_255.jpeg")
