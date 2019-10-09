#!/usr/bin/env python

from skimage import data, io, filters
import matplotlib.pyplot as plt

image = data.coins()

edges = filters.sobel(image)

print(type(image))
print(image.shape)
print(image.size)

io.imshow(edges)
io.show()
