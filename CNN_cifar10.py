#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:32:08 2019

@author: DANG Chen
"""

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import regularizers
from keras.layers.core import Dropout

import matplotlib.pyplot as plt

def plot_history(history, with_val = False):
  history_dict = history.history
  loss_values = history_dict['loss']
  if with_val:
    val_loss_values = history_dict['val_loss']
  epochs = range(1, len(loss_values) + 1)
  plt.figure(figsize=(15,5))
  plt.subplot(1,2,1)
  plt.plot(epochs, loss_values, label='Training loss')
  if with_val:
    plt.plot(epochs, val_loss_values, label='Validation loss')
  plt.title('Training loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  acc = history_dict['acc']
  if with_val:
    val_acc = history_dict['val_acc']
  plt.subplot(1,2,2)
  plt.plot(epochs, acc, label='Training acc')
  if with_val:
    plt.plot(epochs, val_acc, label='Validation acc')
  plt.title('Training acc')
  plt.xlabel('Epochs')
  plt.ylabel('acc')
  plt.legend()
  
  plt.show()

# prepare data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

input_shape = x_train[0].shape

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

# seperate validation set
X_val = x_train[:5000]
partial_X_train = x_train[5000:]

labels_val = y_train[:5000]
partial_labels_train = y_train[5000:]

# create model
model = Sequential()
model.add(Conv2D(32, 3, 3, kernel_regularizer=regularizers.l2(0.0001), border_mode='same', activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Dropout(0.5))
model.add(Conv2D(64, 3, 3, kernel_regularizer=regularizers.l2(0.0001), border_mode='same', activation="relu")) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Dropout(0.5))
model.add(Conv2D(128, 3, 3, kernel_regularizer=regularizers.l2(0.0001), border_mode='same', activation="relu")) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Conv2D(128, 3, 3, kernel_regularizer=regularizers.l2(0.0001), border_mode='same', activation="relu")) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# train model
epochs = 10
history = model.fit(partial_X_train, partial_labels_train, verbose=1, batch_size=256, epochs=epochs, validation_data=(X_val, labels_val))

plot_history(history, with_val=True)
model.evaluate(x_test, y_test)

