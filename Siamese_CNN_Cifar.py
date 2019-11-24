# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:31:28 2019

@author: Joachim Dublineau
"""
from __future__ import print_function
from keras.models import Model, Sequential, load_model
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, \
Activation, Flatten, Lambda
from keras import regularizers
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils import np_utils
import random as rd
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

class Data_Cifar_10:
  def __init__(self):
    self.prepare_data()
    self.group_data()

  def prepare_data(self):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    self.img_rows = x_train.shape[1]
    self.img_cols = x_train.shape[2]
    self.input_shape = (self.img_rows, self.img_cols,3)

    x_train = x_train.reshape(-1, self.img_rows, self.img_cols, 3)
    x_test = x_test.reshape(-1, self.img_rows, self.img_cols, 3)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255.
    x_test /= 255.
    self.x_train = x_train
    self.x_test = x_test
    self.y_train = y_train
    self.y_test = y_test

  def group_data(self):
    self.grouped_data = {}
    self.grouped_test_data = {}
    for i in range(10):
      #self.grouped_data[i] = self.x_train[self.y_train==i]
      #elf.grouped_test_data[i] = self.x_test[self.y_test==i]
      self.grouped_data[i] = self.x_train[np.argwhere(self.y_train==i)[:,0]]
      self.grouped_test_data[i] = self.x_test[np.argwhere(self.y_test==i)[:,0]]

  def get_batch(self, num_list, num_classes, flag='train'):
    target = []
    batch = []
    if flag  == 'train':
      data_ = self.grouped_data
    else:
      data_ = self.grouped_test_data
    num_example = min([data_[i].shape[0] for i in num_list])

    for i in range(num_example*num_classes):
      i1,i2 = np.random.choice(num_list,2, replace=False)
      i_index = np.random.choice(num_example,4, replace=False)
      batch += [[data_[i1][i_index[0]], data_[i1][i_index[1]]]]
      batch += [[data_[i1][i_index[2]], data_[i2][i_index[3]]]]
      target.append(1)
      target.append(0)
    return np.array(batch), np.array(target)

  def get_test_batch(self, test_size, test_target, categ_target, k=10):#k-way one shot learning
    batch = [np.zeros((test_size*k, self.img_rows, self.img_cols,3)) for i in range(2)]
    # batch = []
    target = [] # index of correct category
    num_list = test_target
    categ_list = categ_target
    data_ = self.grouped_data
    for i in range(test_size):
      i1 = np.random.choice(num_list)
      i1_index = np.random.choice(data_[i1].shape[0])
      batch[0][i*k:i*k+k,:] = np.repeat(data_[i1][i1_index][np.newaxis,:], k, axis=0)
      target.append(i1)
      for k_i in range(k):
        i2_index = np.random.choice(data_[categ_list[k_i]].shape[0])
        batch[1][i*k+k_i,:] = data_[categ_list[k_i]][i2_index]
        while categ_list[k_i]==i1 and i2_index == i1_index:
          i2_index = np.random.choice(data_[categ_list[k_i]].shape[0])
          batch[1][i*k+k_i,:] = data_[categ_list[k_i]][i2_index] 
    return batch, target

# --------------------------------------
# Siamese CNN network definition 
# --------------------------------------

def get_siamese_model(input_shape, distance = 'l1'):
   
  # # Network params
      
  conv_depth_1 = 100
  kernel_size_1 = 3
  
  conv_depth_2 = 100 
  kernel_size_2 = 3
  pool_size_2 = 2
    
  conv_depth_3 = 200 
  kernel_size_3 = 3
    
  conv_depth_4 = 200 
  kernel_size_4 = 3
    
  conv_depth_5 = 200 
  kernel_size_5 = 3
  pool_size_5 = 2
    
  hidden_size_1 = 600

  hidden_size_2 = 300
    
  weight_penalty = 0.0001 
  
  left_input = Input(input_shape, name='input1')
  right_input = Input(input_shape, name='input2')

  model = Sequential(name = "CNN_embedded")

  model.add(Conv2D(conv_depth_1, (kernel_size_1, kernel_size_1), 
                   padding='same',
                   input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(Dropout(0.3))
  model.add(Conv2D(conv_depth_2, (kernel_size_2, kernel_size_2), 
                   padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(pool_size_2, pool_size_2)))
  model.add(Dropout(0.3))

  model.add(Conv2D(conv_depth_3, (kernel_size_3, kernel_size_3), 
                   padding='same', 
                   kernel_regularizer=regularizers.l2(weight_penalty)))
  model.add(Activation('relu'))
  model.add(Dropout(0.3))
    
  model.add(Conv2D(conv_depth_4, (kernel_size_4, kernel_size_4), 
                   padding='same',
                   kernel_regularizer=regularizers.l2(weight_penalty)))
  model.add(Activation('relu'))
  model.add(Dropout(0.3))
    
  model.add(Conv2D(conv_depth_5, (kernel_size_5, kernel_size_5), 
                   padding='same', 
                   kernel_regularizer=regularizers.l2(weight_penalty)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(pool_size_5, pool_size_5)))
  model.add(Dropout(0.3))
  
  model.add(Flatten())
  model.add(Dropout(0.3))
  model.add(Dense(hidden_size_1, 
                  kernel_regularizer=regularizers.l2(weight_penalty)))
  model.add(Activation('relu'))
  model.add(Dropout(0.3))

  model.add(Dense(hidden_size_2,
                  kernel_regularizer=regularizers.l2(weight_penalty),
                  activation = 'relu'))
  model.add(Dropout(0.3))

  encoded_l = model(left_input)
  encoded_r = model(right_input)

  if distance == 'l1':
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1,activation='sigmoid')(L1_distance)

    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction,
                        name = "Siamese_CNN")

  else:
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([encoded_l, 
                                                         encoded_r])

    siamese_net = Model([left_input,right_input], distance, name="Siamese_CNN")
  return siamese_net

def contrastive_loss(y_true, y_pred):
  '''Contrastive loss from Hadsell-et-al.'06
  http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
  '''
  margin = 1
  square_pred = K.square(y_pred)
  margin_square = K.square(K.maximum(margin - y_pred, 0))
  return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def accuracy_(y_true, y_pred):
  '''Compute classification accuracy with a fixed threshold on distances.
  '''
  return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def compute_nearest_neighbor(vector, data_set):
     """ 
     This function returns the index of the nearest neighbor in the data_set
     while making sure that this neighbor is similar to vector.
     INPUTS:
     - vector: array of coordinates of the same dimension of the vectors in 
     data_set
     - data_set: array of vectors.
     OUTPUT:
     - nn: index of the nearest_neighbor.
     COMPUTATION TIME:
     Immediate
     """
     mini = 1000000
     nn = 0
     for i in range(data_set.shape[0]):
       elem = data_set[i,:]
       dist = np.linalg.norm(vector - elem)
       if dist != 0.:
         if dist < mini:
           nn = i
           mini = dist
     return nn

def test_model_Cifar(num_classes, model_name, x_other, y_other, 
                     training = False, plot = True):
    print("Testing Siamese Model : {} on Cifar10 ...".format(model_name))
    train_num_list = list(range(num_classes))
    val_num_list = list(range(num_classes))
    test_num_list = list(range(num_classes))
    all_num_list = list(range(10))
    model = get_siamese_model((32,32,3))
    model_name = "SiameseCNN_Cifar.h5"

    # Training

    if (training):
        learning_rate = 0.0001
        batch_size = 32    
        num_epochs = 50
        
        data = Data_Cifar_10()
        batch, targets = data.get_batch(train_num_list, num_classes=num_classes,
                                        flag = 'train')
        val_batch, val_targets = data.get_batch(val_num_list, 
                                                num_classes=num_classes,
                                                flag = 'val')
        input_shape = data.input_shape
        
        #rms = RMSprop()
        opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss=contrastive_loss, optimizer=opt, metrics=[accuracy_])
        model.fit([batch[:, 0], batch[:, 1]], targets, 
                  batch_size=batch_size,
                  epochs=num_epochs, 
                  validation_data=([val_batch[:, 0],val_batch[:, 1]],val_targets), 
                  shuffle = True)
        
        # model.save(model_name)
            
    # Loading the model:
    print("Loading model...")
    if not (training):
        model = load_model(model_name, 
                           custom_objects={'contrastive_loss': contrastive_loss,
                                           "accuracy_" : accuracy_})
        print(model.summary())
    print("Done.")
    
    # Feature Extraction:
     
    print("Feature Extraction for a few examples...")
    sequential_model = model.layers[2]
    intermediate_layer_model = Model(inputs=sequential_model.layers[0].input,
                                     outputs=sequential_model.layers[-1].output)
    
    results = intermediate_layer_model.predict(x_other)
    
    # Extraction of some examples
    nb_examples = 1000
    state = True
    i = 0
    results_per_class = [0]*(10-num_classes)
    while state:
     x = intermediate_layer_model.predict(np.array([x_other[i]]))[0]
     y = y_other[i,0]
     if type(results_per_class[y - num_classes]) == type(0): 
       results_per_class[y - num_classes] = [x]
     else:
       if len(results_per_class[y - num_classes]) < nb_examples:
         results_per_class[y - num_classes].append(x)
       else:
         stop = True
         for elem in results_per_class:
           if type(elem) == type(0): 
             stop = False
             break
           if len(elem) < nb_examples:
             stop = False
             break
         if stop: state = False
     i += 1
    results_per_class = np.array(results_per_class)
    #print(results_per_class.shape)
    print("Done.")
    
    # Ploting the last feature for the new classes with dimension reduction 
    # methods:
    if plot:
        print("Ploting results with PCA and TSNE...")
        N = 3000
        feat_cols = [ 'index'+str(i) for i in range(results.shape[1]) ]
        df = pd.DataFrame(results,columns=feat_cols)
        df['y'] = y_other
        df['label'] = df['y'].apply(lambda i: str(i))
        #print('Size of the dataframe: {}'.format(df.shape))
        
        ########### PCA ##########
        # np.random.seed(42)
        rndperm = np.random.permutation(df.shape[0])
        df_subset = df.loc[rndperm[:N],:].copy()
        data_subset = df_subset[feat_cols].values
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(data_subset)
        df_subset['pca-one'] = pca_result[:,0]
        df_subset['pca-two'] = pca_result[:,1] 
        df_subset['pca-three'] = pca_result[:,2]
        
        print('Explained variation per \
         principal component: {}'.format(pca.explained_variance_ratio_))
        print('Size of the dataframe: {}'.format(df_subset.shape))
        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax.scatter(
           xs=df_subset.loc[rndperm,:]["pca-one"], 
           ys=df_subset.loc[rndperm,:]["pca-two"], 
           zs=df_subset.loc[rndperm,:]["pca-three"], 
           c=df.loc[rndperm,:]["y"], cmap='tab10'
        )
        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three')
        plt.show()
        
        ######## T-SNE Method ########
        data_subset = df_subset[feat_cols].values
        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(data_subset)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
        df_subset['tsne-2d-one'] = tsne_results[:,0]
        df_subset['tsne-2d-two'] = tsne_results[:,1]
        
        plt.figure(figsize=(16,10))
        sns.scatterplot(
           x="tsne-2d-one", y="tsne-2d-two",hue="y",
           palette=sns.color_palette("hls", 10-num_classes),
           data=df_subset, legend="full", alpha=0.3)
        plt.show()
        print("Done.")
        
    # Prediction evaluation on new classes:
     
    # Tests:
    
    print("Computing the confusion matrix for one-shot learning on 300 different references...")
    confusion_matrix = np.zeros((10-num_classes, 10 - num_classes), dtype = int)
    # ligne : prediction
    # colonne : reference
    nb_of_different_references = 300
    accuracies = np.zeros((10-num_classes, nb_of_different_references))
    for p in range(nb_of_different_references):
     index_ref = rd.randint(0, nb_examples-1)
     references = results_per_class[:,index_ref]
     for i in range(10-num_classes):
       nb_mistakes = 0
       k = 0
       for elem in results_per_class[i,:]:
         if k != index_ref:
           index = compute_nearest_neighbor(elem, references)
           confusion_matrix[index, i] += 1
           if index != i:
             nb_mistakes += 1
         k += 1
       accuracies[i, p] = 1 - nb_mistakes/(nb_examples-1)  
    for num_class in range(10-num_classes):
     print("Class n°", num_class+num_classes,"accuracy:", 
           np.mean(accuracies[num_class,:]))
    print("Confusion Matrix:")
    print(confusion_matrix)
    print("Done.")
    
    print("Computing the confusion matrix for 3-shots learning on 100 different references...")
    confusion_matrix = np.zeros((10-num_classes, 10 - num_classes), dtype = int)
    # ligne : prediction
    # colonne : reference
    nb_of_different_references = 100
    accuracies = np.zeros((10-num_classes, nb_of_different_references))
    for p in range(nb_of_different_references):
      index_ref = rd.randint(0, nb_examples-3)
      references = results_per_class[:,index_ref] + results_per_class[:, index_ref +1 ] + \
      results_per_class[:,index_ref+2]
      references /= 3
      for i in range(10-num_classes):
        nb_mistakes = 0
        k = 0
        for elem in results_per_class[i,:]:
          if k != index_ref:
            index = compute_nearest_neighbor(elem, references)
            confusion_matrix[index, i] += 1
            if index != i:
              nb_mistakes += 1
          k += 1
        accuracies[i, p] = 1 - nb_mistakes/(nb_examples-1)  
    for num_class in range(10-num_classes):
      print("Class n°", num_class+num_classes,"accuracy:", 
            np.mean(accuracies[num_class,:]))
    print("Confusion Matrix:")
    print(confusion_matrix)
    
    print("Done. End Testing.")
