#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 18:00:13 2019

@author: luke
"""

from keras.datasets import cifar100
import cv2

#https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
data = cifar100.load_data()
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

for i in range(1000):
    print(y_train[i])
    cv2.imshow("mnist", x_train[i])
    cv2.waitKey(2000)
