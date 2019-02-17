#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:20:22 2019

@author: luke
"""

import cv2

# 读取一张图像
color_img = cv2.imread('1.jpeg')

print(color_img.shape)
#print(type(color_img))
#cv2.imshow("test", color_img)

#cv2.waitKey(10000)

#print(color_img)
color_img[:, :, 0] = 0
color_img[:, :, 2] = 0
print(color_img[0].shape)

cv2.imshow("test", color_img)

cv2.waitKey(10000)