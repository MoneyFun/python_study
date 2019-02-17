#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 18:59:55 2019

@author: luke
"""

#定义一个类，类名为Bird
class Bird():
    #定义构造函数
    def __init__(self):
        #定义类中的变量
        self.hungry = True

    #定义类中的函数
    def eat(self): 
        if self.hungry:
            print('Aaaah...')
            self.hungry = False
        else:
            print('No,thanks!')


b1=Bird()
b1.eat()
b1.eat()
b1.eat()
print(b1.hungry)
print(b1.hungry)