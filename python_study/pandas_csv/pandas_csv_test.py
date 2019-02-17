# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 21:43:19 2019

@author: luke
"""

import pandas as pd
import numpy as np

'''
arr1 = np.arange(100).reshape(10,10)
data1 = pd.DataFrame(arr1)
data1.to_csv('data1.csv')
'''
df = pd.read_csv('data1.csv', index_col = 0, header = None)
print(type(df))
print(df[2])
'''
k = df.as_matrix()
print(type(k))
print(k)

k = np.array(df)
print(type(k))
print(k)
print(k.dtype)
'''