from matplotlib import pyplot as plt

from hyperparameters import build_model, train_ds, val_ds

EPOCHS = 50
LEARNING_RATE = 0.0006425108307104302
DROPOUT_RATE = 0.46386941186999164

model = build_model(LEARNING_RATE, DROPOUT_RATE)
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
model.save('C:/Users/theos/PycharmProjects/FinalYearProject/tuned-trained-model.h5')

## Assigning the history of the model to variables
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

## Plotting the training and validation accuracy and loss
plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()