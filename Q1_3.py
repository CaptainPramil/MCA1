# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:45:32 2020

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:11:46 2020

@author: admin
"""

import numpy as np
from skimage.feature import blob_doh
from math import sqrt
from skimage import data
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from skimage.color import rgb2gray

img =cv2.imread("all_souls_000002.jpg",0)
img= rgb2gray(img)


blobs = [blob_doh(
        img,
        min_sigma=1,
        max_sigma=60,
        num_sigma=10,
        log_scale=True,
        threshold=.005)]


blobs_list = [blobs]
colors = ['yellow']
titles = ['Determinat of Hessian']

sequence = zip(blobs, colors, titles)
fig, A = plt.subplots()
nh,nw = img.shape

for idx, (blobs, color, title) in enumerate(sequence):
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
    
plt.tight_layout()
plt.show()