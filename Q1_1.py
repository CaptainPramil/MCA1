# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:07:22 2020

@author: Pramil
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def unique(a):
    
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis = 0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis = 1)
    return a[ui]

def isValid(X, Y, point):
    """
    Check if point is a valid pixel
    """
    if point[0] < 0 or point[0] >= X:
        return False
    if point[1] < 0 or point[1] >= Y:
        return False
    return True
 
def getNeighbors(X, Y, x, y, dist):
    """
    Find pixel neighbors according to various distances
    """
    All_n = []

    n1 = (x + dist, y + dist)
    n2 = (x + dist, y)
    n3 = (x + dist, y - dist)
    n4 = (x, y - dist)
    n5 = (x - dist, y - dist)
    n6 = (x - dist, y)
    n7 = (x - dist, y + dist)
    n8 = (x, y + dist)
 
    neighbours = (n1, n2, n3, n4, n5, n6, n7, n8)
 
    for i in neighbours:
        if isValid(X, Y, i):
          All_n.append(i)

    return All_n
 
def correlogram(img, Cm, K):
    
    X, Y = img.shape
 
    colorsPercent = []

    for k in K:
        # print "k: ", k
        countColor = 1
 
        color = []
        for i in Cm:
           color.append(0)
 
        for x in range(0, X, int(round(X / 10))):
            for y in range(0, Y, int(round(Y / 10))):

                Ci = img[x][y]
                Cn = getNeighbors(X, Y, x, y, k)
                for j in Cn:
                    Cj = img[j[0]][j[1]]
 
                    for m in range(len(Cm)):
                        if np.array_equal(Cm[m], Ci) and np.array_equal(Cm[m], Cj):
                            countColor = countColor + 1
                            color[m] = color[m] + 1

        for i in range(len(color)):
            color[i] = float(color[i]) / countColor
        
        colorsPercent.append(color)

    return colorsPercent


def autoCorrelogram(img):
    K = 64
    Z = img
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # according to "Image Indexing Using Color Correlograms" paper
    K = [i for i in range(1, 9, 2)]

    colors64 = unique(np.array(res))

    result = correlogram(res2, colors64, K)
    return result

img = cv2.imread("all_souls_000001.jpg",0)

col=autoCorrelogram(img)
fig, A = plt.subplots()
A.imshow(col, interpolation='nearest',cmap="gray")
A.plot()  
plt.show()