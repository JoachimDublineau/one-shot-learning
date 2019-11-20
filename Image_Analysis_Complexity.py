# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:27:35 2019

@author: Joachim Dublineau
"""

########### Image Analysis: Detection of Complexity #############

import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
from PIL import Image

# Few Hyperparameters:
goal_size = (32,32)
image_size = (400,300)
threshold_1 = 3.5
threshold_2 = 4.0


# Load Images & lowering the quality:

images = []
for i in range(7):
    img = Image.open("img{}.jpg".format(i + 1))
    img = img.resize(image_size,Image.ANTIALIAS)
    img = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
    images.append(img)

def entropy_full(signal):
        '''
        function returns entropy of a signal
        signal must be a 1-D numpy array
        '''
        lensig=signal.size
        symset=list(set(signal))
        propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
        ent=np.sum([p*np.log2(1.0/p) for p in propab])
        return ent

"""genre potentiellement, je peux faire après cette image d'entropy, 
un calcul d'objet en mode délimitation de zone à forte entropie,
ca me donne une idée du nombre d'objet et de leur taille donc je 
peux choisir la taille de ma fenetre"""

"""pour l'instant on part sur un calcul de seuil basé sur la première entropie
calculée."""

def determine_size_sliding_window(entropy, thres1 = threshold_1, 
                                  thres2 = threshold_2):
    """ Returns the shape of the sliding window depending on the entropy. """
    if entropy < thres1:
        return goal_size[0]*4, goal_size[0]*4
    else:
        if entropy < thres2:
           return goal_size[0]*3, goal_size[0]*3
        else:
           return goal_size[0]*2, goal_size[0]*2
offset_x, offset_y = 50,50

for img in images: 
    entropy_img = entropy(img[:,:,0], disk(10))
    entropy_ = np.mean(entropy_img)
#    plt.imshow(entropy_img)
#    plt.show()
    print(np.mean(entropy_))
    x_size, y_size = determine_size_sliding_window(entropy_)
    for i in range(x_size):
        for j in range(y_size):
            img[offset_x + i, offset_y + j, 0] += 60 
    plt.imshow(img)
    plt.show()
    #print(entropy_full(np.reshape(img, -1)))
    
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            
def extract_image_from_sliding_window(image, ref_point, size_window, 
                                      goal_size = goal_size):
    """Returns the array of the image inside the sliding window."""
    square_size = (int(size_window[0]/goal_size[0]), int(size_window[1]/goal_size[1]))
    extracted_image = np.zeros(goal_size, dtype = np.float32)
    for i in range(goal_size[0]):
        for j in range(goal_size[1]):
            extracted_image[i,j] = np.mean(image[square_size[0]*i+ref_point[0]: \
                                           square_size[0]*(i+1)+ref_point[0]+1, 
                                           square_size[1]*j+ref_point[1]:\
                                           square_size[1]*(j+1)+ref_point[1]+1,:])
    return None

for (x, y, window) in sliding_window(images[0], stepSize=32, windowSize=(64, 64)):
    print(x, y, window.shape)
    # Here make the prediction with the network 
