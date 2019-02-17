#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:08:34 2019

@author: luke
"""

import numpy as np
# 将数组以二进制格式保存到磁盘
arr=np.array([1,2,3,4,5])
np.save('test',arr)
# 读取数组
print(np.load('test.npy'))
