#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 09:28:51 2019

@author: luke
"""

import pandas as pd

data = [{'a,"': 1, 'b': 2},{'a,"': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
print df
df.to_csv('out.csv')
df2 = pd.read_csv('out.csv',index_col = 0)
print(df2)