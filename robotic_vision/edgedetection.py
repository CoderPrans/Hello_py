#!/usr/bin/env python

import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage import io, color, exposure, data
from pylab import subplot

# Load the image
img = data.astronaut()              

# Convert the image to grayscale (1 channel)
img = color.rgb2gray(img)       

subplot(1,2,1)
plt.imshow(img, cmap=plt.cm.gray)
plt.axis('off')

# edgeDetection kernal
kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

# we use 'valid' which means we do not add zero padding to our image
edges = scipy.signal.convolve2d(img, kernel, 'valid')

print('\n First 5 columns and rows of the edges matrix: \n', edges[:5,:5]*255)

# Adjust the contrast of the filtered image by applying Histogram Equalization
edges_equalized = exposure.equalize_adapthist(edges/np.max(np.abs(edges)), clip_limit=0.03)

subplot(1,2,2)
plt.imshow(edges_equalized, cmap=plt.cm.gray)    # plot the edges_clipped
plt.axis('off')
plt.show()
