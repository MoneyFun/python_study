#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:36:43 2019

@author: luke
"""

import numpy as np
import cv2

data = np.load('mnist.npz')
print(data)
x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

for i in range(100):
    print(y_train[i])
    cv2.imshow("mnist", x_train[i])
    cv2.waitKey(1000)