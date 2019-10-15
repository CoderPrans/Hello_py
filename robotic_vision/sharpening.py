#!/usr/bin/env python

import numpy as np
import scipy
from skimage import io, color, exposure, data
from pylab import subplot
import matplotlib.pyplot as plt

# Load the image
img = data.astronaut()              

# Convert the image to grayscale (1 channel)
img = color.rgb2gray(img)       

subplot(1,2,1)
plt.imshow(img, cmap=plt.cm.gray)
plt.axis('off')

# sharpening kernal
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

# you can use 'valid' instead of 'same', then it will not add zero padding
image_sharpen = scipy.signal.convolve2d(img, kernel, 'same')

print('\n First 5 columns and rows of the image_sharpen matrix: \n', image_sharpen[:5,:5]*255)

# Adjust the contrast of the filtered image by applying Histogram Equalization 
image_sharpen_equalized = exposure.equalize_adapthist(image_sharpen/np.max(np.abs(image_sharpen)), clip_limit=0.03)

subplot(1,2,2)
plt.imshow(image_sharpen_equalized, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
