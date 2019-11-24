# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:30:13 2019

@author: Joachim Dublineau
"""

from Siamese_CNN_Cifar import test_model_cifar
from keras.datasets import cifar10
import numpy as np
from keras.utils import np_utils

num_classes = 5
model_name = "SiameseCNN_Cifar.h5"

def load_cifar_10(num_classes):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') 
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.
    
    
    # Extraction of the train data with num_classes != 10:
    
    x_train_bis, y_train_bis = [], []
    x_test_bis, y_test_bis = [], []
    x_other, y_other = [], []
    
    if num_classes != 10:
      for i in range(x_train.shape[0]):
        if y_train[i,0] < num_classes:
          x_train_bis.append(x_train[i])
          y_train_bis.append(y_train[i])
        else:
          x_other.append(x_train[i])
          y_other.append(y_train[i])
      for i in range(x_test.shape[0]):
        if y_test[i,0] < num_classes:
          x_test_bis.append(x_test[i])
          y_test_bis.append(y_test[i])
        else:
          x_other.append(x_test[i])
          y_other.append(y_test[i])
    
    x_train_bis = np.array(x_train_bis, dtype = np.float32)
    y_train_bis = np.array(y_train_bis, dtype = int)
    x_test_bis = np.array(x_test_bis, dtype = np.float32)
    y_test_bis = np.array(y_test_bis, dtype = int)
    x_other = np.array(x_other, dtype = np.float32)
    y_other = np.array(y_other, dtype = int)
    
    y_train_bis = np_utils.to_categorical(y_train_bis, num_classes) # One-hot encode the labels
    y_test_bis = np_utils.to_categorical(y_test_bis, num_classes)
    y_train_bis = y_train_bis.astype('int')
    y_test_bis = y_test_bis.astype('int')
    return x_train_bis, y_train_bis, x_test_bis, y_test_bis, x_other, y_other 

x_train_bis, y_train_bis, x_test_bis, y_test_bis, x_other, y_other = \
load_cifar_10(num_classes)

test_model_cifar(num_classes, model_name, x_other, y_other, 
                     training = False, plot = True)