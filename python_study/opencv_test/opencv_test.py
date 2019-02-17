# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 11:57:21 2019

@author: luke
"""

import cv2
'''
# 读取一张图像
color_img = cv2.imread('1.jpeg')

print(color_img.shape)
print(type(color_img))


color_img[300][250][0] = 0
color_img[300][250][1] = 255
color_img[300][250][2] = 0

color_img[300, :, 0] = 0
color_img[300, :, 1] = 0
color_img[300, :, 2] = 255


cv2.imshow("test", color_img)

cv2.waitKey(0)

'''
# 直接读取单通道灰度图
gray_img = cv2.imread('1.jpeg', cv2.IMREAD_GRAYSCALE)
print(gray_img.shape)
print(type(gray_img))

#cv2.imshow("gray_img", gray_img)

#cv2.waitKey(10000)

cv2.imwrite("write_img.jpg", gray_img)
newImg = cv2.imread("write_img.jpg")
print(newImg.shape)

cv2.imshow("newImg", newImg)
cv2.waitKey(10000)

