#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:35:09 2019

@author: luke
"""

import cv2

# 读取一张图像
color_img = cv2.imread('1.jpeg')

print(color_img.shape)
print(type(color_img))
#cv2.imshow("test", color_img)

#cv2.waitKey(10000)

# cv2.IMWRITE_JPEG_QUALITY指定jpg质量，范围0到100，默认95，越高画质越好，文件越大
cv2.imwrite('test_imwrite.jpg', color_img, (cv2.IMWRITE_JPEG_QUALITY, 10))

# cv2.IMWRITE_PNG_COMPRESSION指定png质量，范围0到9，默认3，越高文件越小，画质越差
cv2.imwrite('test_imwrite.png', color_img, (cv2.IMWRITE_PNG_COMPRESSION, 1))