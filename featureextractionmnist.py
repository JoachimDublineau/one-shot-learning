# -*- coding: utf-8 -*-
"""
SimplestFeatureExtractionMNIST
Show 2d or 3d features extracted with a shallow dense NN

Created on Sun Nov 10 20:05:58 2019
@author: DANG
"""

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dense, Reshape
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###################################
# Data Preparation
###################################
def prepare_data():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  img_rows = x_train.shape[1]
  img_cols = x_train.shape[2]

  x_train = x_train.reshape(-1, img_rows, img_cols, 1)
  x_test = x_test.reshape(-1, img_rows, img_cols, 1)
  x_train = x_train.astype("float32")
  x_test = x_test.astype("float32")
  x_train /= 255
  x_test /= 255
  return x_train, y_train, x_test, y_test

###################################
# Model Preparation
###################################
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

  acc = history_dict['accuracy']
  if with_val:
    val_acc = history_dict['val_accuracy']
  plt.subplot(1,2,2)
  plt.plot(epochs, acc, label='Training acc')
  if with_val:
    plt.plot(epochs, val_acc, label='Validation acc')
  plt.title('Training accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('accuracy')
  plt.legend()
  
  plt.show()

def get_dense_model():
  model = Sequential()
  model.add(Reshape((28*28,)))
  model.add(Dense(512, input_shape=(None, 28*28), activation = 'relu'))
  model.add(Dense(128, activation = 'relu'))
  model.add(Dense(3, activation = 'relu'))
  model.add(Dense(10, activation = 'softmax'))

  model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  return model

def train_model(model, x_train, y_train):
  epochs = 10
  history = model.fit(x_train, y_train, verbose=1, batch_size=64,  epochs=epochs)
  plot_history(history)
  return model

def get_feature_model(model):
  feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
  return feature_model

def get_feature(model, data):
  features = model.predict(data)
  return features

###################################
# Feature Plotting
###################################
def plot_scatter_cato(features, labels, list_active_labels):
  fig, ax = plt.subplots()
  ax.margins(0.05)
  for i in list_active_labels:
    index = [labels[:,i]==1]
    ax.plot(features[index][0], features[index][1], marker='o', linestyle='', ms=12, label=i)
  ax.legend()
  plt.show()

def plot_scatter3d_cato(features, labels, list_active_labels):
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.margins(0.05)
  for i in list_active_labels:
    index = [labels[:,i]==1]
    ax.plot(features[index][0], features[index][1], features[index][2], marker='o', linestyle='', ms=12, label=i)
  ax.legend()
  plt.show()

def scatter_all(features, labels):
  plt.scatter(features[:,0], features[:,1], c=labels)
  plt.show()

def scatter3d_all(features, labels):
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter(features[:,0], features[:,1], features[:,2], c=labels)
  plt.show()

if __name__ == '__main__':
  x_train, y_train, x_test, y_test = prepare_data()
  y_train_cato = to_categorical(y_train)
  y_test_cato = to_categorical(y_test)

  model = get_dense_model()
  model = train_model(model, x_train, y_train_cato)
  model.evaluate(x_test, y_test_cato)

  f_model = get_feature_model(model)
  features = get_feature(f_model, x_train)

  plot_scatter3d_cato(features, y_train_cato, list(range(10)))

  scatter3d_all(features, y_train)