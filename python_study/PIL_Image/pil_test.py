# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 08:29:23 2019

@author: luke
"""

from PIL import Image
sourceFileName = "source.png"
avatar = Image.open(sourceFileName)
#avatar.show()
#sudo apt-get install imagemagic
print(avatar.format, avatar.size, avatar.mode)

'''
avatar.thumbnail((50, 50))
avatar.save("outfile_50.jpg", "JPEG")
print(avatar.size)

'''

resizedIm = avatar.resize((10, 10))
resizedIm.save("outfile_10.jpg", "JPEG")
print(resizedIm.size)
print(avatar.size)
