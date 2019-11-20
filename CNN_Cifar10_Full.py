# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 22:52:23 2019

@author: Joachim Dublineau
"""
# Importations:
from __future__ import print_function
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.optimizers import RMSprop, Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, \
Flatten, Conv3D, MaxPooling3D
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import random as rd

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.

# Learning params:

num_classes = 6
learning_rate = 0.0001
batch_size = 32    
num_epochs = 20 

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

print("Train set:", x_train_bis.shape[0], "images")
print("Test set:", x_test_bis.shape[0], "images")
print("X Other classes:", x_other.shape[0])

y_train_bis = np_utils.to_categorical(y_train_bis, num_classes) # One-hot encode the labels
y_test_bis = np_utils.to_categorical(y_test_bis, num_classes)
y_train_bis = y_train_bis.astype('int')
y_test_bis = y_test_bis.astype('int')

# --------------------------------------
# CNN network definition 
# --------------------------------------

# Network params
  
conv_depth_1 = 100
kernel_size_1 = 3

conv_depth_2 = 100 
kernel_size_2 = 3
pool_size_2 = 2

conv_depth_3 = 200 
kernel_size_3 = 3

conv_depth_4 = 200 
kernel_size_4 = 3

conv_depth_5 = 400 
kernel_size_5 = 3
pool_size_5 = 2

hidden_size_1 = 600

weight_penalty = 0.0001 


model = Sequential()

model.add(Conv2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same',
                 input_shape=x_train_bis.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size_2, pool_size_2)))
model.add(Dropout(0.3))

model.add(Conv2D(conv_depth_3, (kernel_size_3, kernel_size_3), padding='same', 
                 kernel_regularizer=regularizers.l2(weight_penalty)))
model.add(Activation('relu'))

model.add(Conv2D(conv_depth_4, (kernel_size_4, kernel_size_4), padding='same',
                 kernel_regularizer=regularizers.l2(weight_penalty)))
model.add(Activation('relu'))

model.add(Conv2D(conv_depth_5, (kernel_size_5, kernel_size_5), padding='same', 
                 kernel_regularizer=regularizers.l2(weight_penalty)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size_5, pool_size_5)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(hidden_size_1, 
                kernel_regularizer=regularizers.l2(weight_penalty)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model_name = "test2.h5"

# # Training on whole dataset

# opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
# #opt = RMSprop(lr=learning_rate,decay=1e-6)

# model.compile(loss='categorical_crossentropy', 
#                optimizer=opt, metrics=['accuracy'])
# history = model.fit(x_train_bis, y_train_bis, verbose = True, 
#                     epochs = num_epochs, batch_size=batch_size,
#                     validation_split=0.2, shuffle = True)

## Ploting the loss evolution during training
#loss = history.history["loss"]
#val_loss = history.history["val_loss"]
#epochs = range(1, len(loss) + 1)
#plt.plot(epochs, loss, "bo", label="Training loss")
#plt.plot(epochs, val_loss, "b", label="Validation loss")
#plt.title("Training and validation loss")
#plt.xlabel("Epochs")
#plt.ylabel("Loss")
#plt.legend()
#plt.show()

# model.save(model_name)

model = load_model(model_name)

print(model.summary())

# Test

print("Performance on train:", model.evaluate(x_train_bis, y_train_bis))
print("Performance on test:", model.evaluate(x_test_bis, y_test_bis))

##### FEATURE EXTRACTION #####


intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[-3].output)

results = intermediate_layer_model.predict(x_other)
print(results.shape)
print(y_other.shape)

# Extraction of some examples
nb_examples = 700
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
print(results_per_class.shape)

# Ploting the last feature for the new classes with dimension reduction 
# methods:

N = 7000
feat_cols = [ 'index'+str(i) for i in range(results.shape[1]) ]
df = pd.DataFrame(results,columns=feat_cols)
df['y'] = y_other
df['label'] = df['y'].apply(lambda i: str(i))
print('Size of the dataframe: {}'.format(df.shape))

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

# # Test compute_nearest_neighbor:

# data_set = [[0,0], [1,4], [1,1]]
# data_set = np.array(data_set)
# vector = np.array([0,0])
# print(compute_nearest_neighbor(vector, data_set))

# Prediction evaluation on new classes:
  
# Tests:
confusion_matrix = np.zeros((10-num_classes, 10 - num_classes), dtype = int)
# ligne : prediction
# colonne : reference
nb_of_different_references = 100
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