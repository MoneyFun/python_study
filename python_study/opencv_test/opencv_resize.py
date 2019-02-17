#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:41:37 2019

@author: luke
"""

#导入cv模块
import cv2
# 读取一张原始图片
img = cv2.imread('1.jpeg')
print(img)
'''
patch_img = img[220:550, -180:-50]
# 缩放成200x200的方形图像
#img_200x200 = cv2.resize(img, (200, 200))
#cv2.imshow("img_200x200", img_200x200)
#cv2.waitKey(10000)

# 不直接指定缩放后大小，通过fx和fy指定缩放比例，0.5则长宽都为原来一半
# 等效于img_100x100 = cv2.resize(img, (100, 100))，注意指定大小的格式是(宽度,高度)
# 插值方法默认是cv2.INTER_LINEAR，这里指定为最近邻插值
img_100x100 = cv2.resize(img_200x200, (0, 0), fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_NEAREST)
#cv2.imshow("img_100x100", img_100x100)
#cv2.waitKey(10000)

# 在上张图片的基础上，上下各贴50像素的黑边，生成300x300的图像
img_200x100 = cv2.copyMakeBorder(img_100x100, 50, 50, 0, 0,
                                 cv2.BORDER_CONSTANT,value=(0, 0, 0))

cv2.imshow("img_200x100", img_200x100)
cv2.waitKey(10000)

'''
# 对照片中局部进行剪裁
patch_img = img[220:550, -180:-50, :]
cv2.imshow("patch_img", patch_img)
cv2.waitKey(10000)
#print(patch_img)
'''
cv2.imwrite('img/cropped_img.jpg', patch_img)
cv2.imwrite('img/resized_200x200.jpg', img_200x200)
cv2.imwrite('img/resized_100x100.jpg', img_100x100)
cv2.imwrite('img/bordered_200x100.jpg', img_200x100)
'''