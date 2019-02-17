# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 22:43:53 2019

@author: luke
"""

import numpy as np
data=np.array([1,2,3,4,5], dtype=np.int8)
print(data.dtype)
np.savetxt("data.txt",data, fmt='%d')
#np.save("data.txt",data)
