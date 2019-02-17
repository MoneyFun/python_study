#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:17:48 2019

@author: luke
"""

from keras.datasets import mnist
import cv2

#https://s3.amazonaws.com/img-datasets/mnist.npz
data = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

for i in range(1000):
    print(y_train[i])
    cv2.imshow("mnist", x_train[i])
    cv2.waitKey(2000)