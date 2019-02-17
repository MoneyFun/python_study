# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 11:28:09 2019

@author: luke
"""

from PIL import Image, ImageFilter
im_path = "timg.jpeg"

im = Image.open(im_path)
#im.show()

print(im.size)
cropedIm = im.crop((600, 0, 1800, 800))
cropedIm.save('cropped.png')

im.rotate(90).save('rotate90.png')
im.rotate(270).save('rotate270.png')
im.rotate(180).save('rotate180.png')
im.rotate(20).save('rotate20.png')

im.transpose(Image.FLIP_LEFT_RIGHT).save('transepose_lr.png')
im.transpose(Image.FLIP_TOP_BOTTOM).save('transepose_tb.png')

'''
newIm = im.filter(ImageFilter.GaussianBlur)

newIm = im.filter(ImageFilter.BLUR)
newIm = im.filter(ImageFilter.EDGE_ENHANCE)
newIm = im.filter(ImageFilter.EMBOSS)
newIm = im.filter(ImageFilter.CONTOUR)
newIm = im.filter(ImageFilter.SHARPEN)
newIm = im.filter(ImageFilter.SMOOTH)
newIm = im.filter(ImageFilter.DETAIL)

im.show()
newIm.show()
'''