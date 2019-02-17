# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 22:23:16 2019

@author: luke
"""
'''
f = open("data.txt","r")
str = f.read()
f.close()
print(str)
print(type(str))


f = open("data.txt","r")
line = f.readline()
#line = line[:-1]
while line:
    print(line)
    line = f.readline()
    #line = line[:-1]
f.close()

data = []
for line in open("data.txt","r"):
    data.append(line)

for s in data:
    print(s[:-1])

'''
f = open("data.txt","r")
data = f.readlines()
f.close()

for s in data:
    print(s[:-1])
