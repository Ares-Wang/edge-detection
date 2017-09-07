import os

indices = 0,3
somelist = [1,4,56,6,8,7]
somelist = [i for j, i in enumerate(somelist) if j not in indices]


import cv2
import numpy as np
from matplotlib import pyplot as plt

if os.name == 'nt':
    img = cv2.imread('..\\BSR\\BSDS500\\data\\images\\test\\43051.jpg',0)
elif os.name == 'posix':
    img = cv2.imread('../BSR/BSDS500/data/images/test/43051.jpg', 0)
else:
    raise 'Unsupported OS'

edges = cv2.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()