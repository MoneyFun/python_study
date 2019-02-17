#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 12:38:36 2019

@author: luke
"""

import numpy as np
import cv2

img = np.zeros((100, 100,3))

partImg = img[10:20, 30:-1, :]
print(partImg.shape)
partImg[:, :, 0] = 30
partImg[:, :, 1] = 30
partImg[:, :, 2] = 200

#print(img)

#cv2.imwrite("new.jpg", img)
cv2.imwrite("partImg.jpg", partImg)
