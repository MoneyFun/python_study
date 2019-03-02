#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:51:44 2019

@author: luke
"""

from keras.datasets import cifar10
import cv2

#https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
data = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

for i in range(1000):
    print(y_train[i])
    cv2.imshow("cifar10", x_train[i])
    cv2.waitKey(1000)

cv2.destroyAllWindows()
