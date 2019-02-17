# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 21:21:25 2019

@author: luke
"""

import pandas as pd

'''
df = pd.DataFrame()
print df

#list
data = [1,2,3,4,5]
print(type(data))
df = pd.DataFrame(data)
print df
print(type(df))
'''
#list
data = [['Alex',10],['Bob',12],['Clarke',13]]
print(type(data))
df = pd.DataFrame(data,columns=['Name','Age'])
print df
print(type(df))


#dict
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}

print(type(data))
df = pd.DataFrame(data)
print df
print(type(df))
'''
#index
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data, index=['rank1','rank2','rank3','rank4'])
print df

#list of dict
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
print(type(data))
print(type(data[0]))
df = pd.DataFrame(data)
print df
print(type(df))
'''

data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data, index=['first', 'second'])
print df