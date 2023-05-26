import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, Resizing
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from bayes_opt import BayesianOptimization

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

def build_model(learning_rate, dropout_rate):
    model = Sequential([
        Rescaling(1./255, input_shape=(512, 512, 3)),
        Resizing(128, 128),
        Conv2D(32, 3, padding='same', activation='relu'),
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

    # Compile the model with the specified learning rate and dropout rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # Set the dropout rate for applicable layers
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = dropout_rate

    return model

# Define the function to optimize
def optimize_model(learning_rate, dropout_rate):
    # Build the model
    model = build_model(learning_rate, dropout_rate)

    # Train the model
    history = model.fit(train_ds, validation_data=val_ds, epochs=50, verbose=0)

    # Get the best validation accuracy
    best_val_accuracy = max(history.history['val_accuracy'])

    return best_val_accuracy


# Define the hyperparameter search space
hyperparameter_space = {
    'learning_rate': (0.001, 0.01),
    'dropout_rate': (0.0, 0.5)
}

# Perform Bayesian optimization
optimizer = BayesianOptimization(f=optimize_model, pbounds=hyperparameter_space, verbose=2)
optimizer.maximize(init_points=3, n_iter=10)

# Get the best hyperparameters and the corresponding validation accuracy
best_hyperparameters = optimizer.max['params']
best_val_accuracy = optimizer.max['target']

print("Best Hyperparameters:")
print(best_hyperparameters)
print("Best Validation Accuracy:")
print(best_val_accuracy)