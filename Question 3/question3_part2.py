import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

# the data, shuffled and split between train and test sets
data= tf.keras.datasets.mnist.load_data()
(X_train, Y_train), (X_test, Y_test) = data

# input image dimensions
img_rows, img_cols = 28, 28

# Preprocess the data

# Reshaping of data for deep learning using Keras
# 2 dimensions per example representing a greyscale image 28x28.
X_train = X_train.reshape(X_train.shape[0], img_rows,img_cols, 1)
X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print(X_train.shape)

# create a validation set
X_valid, X_train = X_train[:5000] / 255.0, X_train[5000:] / 255.0
y_valid, y_train = Y_train[:5000], Y_train[5000:]

# alexnet
# Create a model using Keras Sequential API
model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation="relu", padding="same",
                        input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same",),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same",),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same",),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same",),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
])
# Specify the training configuration (optimizer, loss, metrics)
model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd", metrics=["accuracy"])
# Train the the entire dataset for a given number of "epochs"
# monitoring validation loss and metrics at the end of each epoch
history = model.fit(X_train, y_train,batch_size=64, epochs=3,validation_data=(X_valid, y_valid))

# The returned "history" object holds a record
# of the loss values and metric values during training
print('\nhistory dict:', history.history)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(X_test, Y_test, batch_size=128)
print('test loss, test acc:', results)


