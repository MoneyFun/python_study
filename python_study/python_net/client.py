#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 21:24:23 2019

@author: luke
"""

import socket               # 导入 socket 模块

s = socket.socket()         # 创建 socket 对象
host = socket.gethostname() # 获取本地主机名
port = 12345                # 设置端口号

s.connect((host, port))
print s.recv(1024)
s.close()