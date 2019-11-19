# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:27:35 2019

@author: Joachim Dublineau
"""

########### Image Analysis: Detection of Complexity #############

# Load Images & lowering the quality

import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
from PIL import Image

images = []
for i in range(4):
    img = Image.open("img{}.jpg".format(i + 1))
    img = img.resize((400,300),Image.ANTIALIAS)
    img = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
    images.append(img)
print(images[0].shape)

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

for img in images:
    
    # Ploting image:

    plt.imshow(img)
    plt.show()
    
    # Test with entropy calculation:
    entropy0 = entropy(img[:,:,0], disk(10))
    plt.imshow(entropy0)
    plt.show()
#    entropy1 = entropy(img[:,:,1], disk(10))
#    plt.imshow(entropy1)
#    plt.show()
#    entropy2 = entropy(img[:,:,2], disk(10))
#    plt.imshow(entropy2)
#    plt.show()

    print(entropy_full(np.reshape(img, -1)))

"""genre potentiellement, je peux faire après cette image d'entropy, 
un calcul d'objet en mode délimitation de zone à forte entropie,
ca me donne une idée du nombre d'objet et de leur taille donc je 
peux choisir la taille de ma fenetre"""
