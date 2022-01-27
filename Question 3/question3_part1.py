# import os
# os.kill(os.getpid(),9)

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib as plt

# the data, shuffled and split between train and test sets
data=tf.keras.datasets.mnist.load_data()
(X_train, Y_train), (X_test, y_test) = data

#  create a validation set
X_valid, X_train = X_train[:5000] / 255.0, X_train[5000:] / 255.0
y_valid, y_train = Y_train[:5000], Y_train[5000:]

# Create a model using Keras َ API
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(50, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# Specify the training configuration (optimizer, loss, metrics)
model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd", metrics=["accuracy"])

# Fit model on training data
# Train the the entire dataset for a given number of "epochs"
history = model.fit(X_train, y_train,batch_size=64, epochs=3,validation_data=(X_valid, y_valid))

# The returned "history" object holds a record
# of the loss values and metric values during training
print('\nhistory dict:', history.history)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(X_test, y_test, batch_size=128)
print('test loss, test acc:', results)


