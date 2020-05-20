# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:50:41 2020

@author: admin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import math
from math import sqrt


img = cv2.imread("all_souls_000002.jpg",0)
#img = rgb2gray(img)

img = img/255.00
sigma = 1.1
k = 1.414

def LoG(img,sig):
    #window size 
    log_images = [] #to store responses
    for i in range(1,10):
        y = math.pow(k,i)        
        sigma = sig*y #sigma 
        n = sigma*6
        y,x = np.ogrid[-n//2:n//2+1,-n//2:n//2+1]
        log_filter = (-(2*sigma**2) + (x*x + y*y) ) *  (np.exp(-(x**2/(2.*sigma**2)))*np.exp(-(y**2/(2.*sigma**2)))) * (1/(2*np.pi*sigma**4))
        image  = cv2.filter2D(img,-1, log_filter)# convolving image
        image = np.pad(image,((1,1),(1,1)),'constant') #padding 
        image = np.square(image) # squaring the response
        log_images.append(image)
    log_image_np = np.array([i for i in log_images]) # storing the #in numpy array
    return log_image_np

log_image_np = LoG(img,sigma)


def blob_overlap(blob1, blob2):
    n_dim = len(blob1) - 1
    rndim = sqrt(n_dim)

    r1 = blob1[-1] * rndim
    r2 = blob2[-1] * rndim
    
    d = sqrt(np.sum((blob1[:-1] - blob2[:-1])**2))
    
    if d > r1 + r2:
        return 0
    elif d <= abs(r1 - r2):
        return 1
    else:
        R1 = (d**2 + r1**2 - r2**2) / (2*d* r1)
        R1 = np.clip(R1, -1, 1)
        acos1 = math.acos(R1)

        R2 = (d**2 + r2**2 - r1**2) / (2*d*r2)
        R2 = np.clip(R2, -1, 1)
        acos2 = math.acos(R2)

        a = -d + r2 + r1
        b = d - r2 + r1
        c = d + r2 - r1
        d = d + r2 + r1

        A = (r1**2*acos1 + r2**2*acos2 -0.5*sqrt(abs(a*b*c*d)))
        return A/(3.14 * (min(r1, r2)**2))
   
 
def redundancy(bl_rr, olp):
    sigma = bl_rr[:, -1].max()
    distance = 2 * sigma * sqrt(bl_rr.shape[1] - 1)
    tree = spatial.cKDTree(bl_rr[:, :-1])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return bl_rr
    else:
        for (i, j) in pairs:
            blob1, blob2 = bl_rr[i], bl_rr[j]
            n_dim = len(blob1) - 1
    rndim = sqrt(n_dim)

    r1 = blob1[-1] * rndim
    r2 = blob2[-1] * rndim
    
    d = sqrt(np.sum((blob1[:-1] - blob2[:-1])**2))
    
    if d > r1 + r2:
        return 0
    elif d <= abs(r1 - r2):
        return 1
    else:
        R1 = (d**2 + r1**2 - r2**2) / (2*d* r1)
        R1 = np.clip(R1, -1, 1)
        acos1 = math.acos(R1)

        R2 = (d**2 + r2**2 - r1**2) / (2*d*r2)
        R2 = np.clip(R2, -1, 1)
        acos2 = math.acos(R2)

        a = -d + r2 + r1
        b = d - r2 + r1
        c = d + r2 - r1
        d = d + r2 + r1

        A = (r1**2*acos1 + r2**2*acos2 -0.5*sqrt(abs(a*b*c*d)))
        if A > olp:
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0

    return np.array([b for b in bl_rr if b[-1] > 0])

def detect_blob(log_image_np):
    co_ordinates = []
    (h,w) = img.shape
    for i in range(1,h):
        for j in range(1,w):
            slice_img = log_image_np[:,i-1:i+2,j-1:j+2]
            result = np.amax(slice_img)
            if result >= 0.03:
                z,x,y = np.unravel_index(slice_img.argmax(),slice_img.shape)
                co_ordinates.append((i+x-1,j+y-1,k**z*sigma))
    return co_ordinates

co_ordinates = list(set(detect_blob(log_image_np)))
co_ordinates = redundancy(np.array(co_ordinates),0.5)

fig, A = plt.subplots()
nh,nw = img.shape
A.imshow(img, interpolation='nearest',cmap="gray")
for blob in co_ordinates:
    y,x,r = blob
    c = plt.Circle((x, y), r*k, color='red', linewidth=1.5, fill=False)
    A.add_patch(c)
A.plot()  
plt.show()