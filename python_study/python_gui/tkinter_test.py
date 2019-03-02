#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 21:27:19 2019

@author: luke
"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
from Tkinter import *
import Tkinter
import tkMessageBox
from PIL import Image,ImageTk

top = Tkinter.Tk()

# 创建一个Canvas，设置其背景色为白色
cv = Tkinter.Canvas(top, width=750, height=500, bg = 'white')

image = Image.open("1.jpeg")
im = ImageTk.PhotoImage(image)

cv.create_image(100,100,image = im, anchor = NW)
'''
# 创建一个矩形，坐标为(10,10,110,110)
cv.create_rectangle(10,10,110,110)
'''
#cv.pack()

L1 = Label(top, text="网站名")
L1.pack( side = Tkinter.LEFT)
E1 = Entry(top, bd =5)
E1.pack(side = Tkinter.LEFT)


def helloCallBack():
   tkMessageBox.showinfo( "Hello Python", E1.get())
B = Tkinter.Button(top, text ="点我", command = helloCallBack)
B.pack()


top.mainloop()