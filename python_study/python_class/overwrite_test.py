#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 19:18:59 2019

@author: luke
"""

class Person(object):   # 定义一个父类
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.weight = 'weight'

    def talk(self):    # 父类中的方法
        print("person is talking....")

class Chinese(Person):    # 定义一个子类， 继承Person类
    def __init__(self, name, age, language):
        super(Chinese, self).__init__(name, age)
        self.language = language

    def walk(self):      # 在子类中定义其自身的方法
        print('is walking...')


    def talk(self):
        print("speak chinese ...")

c = Chinese('xiaoming', 18, "Chinese")
c.talk()      # 调用继承的Person类的方法
c.walk()     # 调用本身的方法
print(c.weight)
print(c.language)
