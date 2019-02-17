#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:48:19 2019

@author: luke
"""

import wave
import pylab as pl
import numpy as np

#首先载入Python的标准处理WAV文件的模块，然后调用wave.open打开wav文件，注意需要使用"rb"(二进制模式)打开文件：
f = wave.open(r"xiaodaohuixuqu.wav", "rb")
#f = wave.open(r"sweep.wav", "rb")
# 读取格式信息
# (nchannels, sampwidth, framerate, nframes, comptype, compname)
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
#通道数， 采样位宽， 帧率， 总帧数（时常）
print(nchannels, sampwidth, framerate, nframes)

# 读取波形数据
str_data = f.readframes(nframes)
print(type(str_data))
#print(str_data)
#readframes：读取声音数据，传递一个参数指定需要读取的长度，readframes返回的是二进制数据（一大堆bytes)，在Python中用字符串表示二进制数据：
f.close()

#将波形数据转换为数组
#接下来需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组：
wave_data = np.fromstring(str_data, dtype=np.short)
print(type(wave_data))
print(wave_data.shape)
print(wave_data.dtype)

#for i in range(100):
#    print(wave_data[44100 * 10 + i])

#通过fromstring函数将字符串转换为数组，通过其参数dtype指定转换后的数据格式，由于我们的声音格式是以两个字节表示一个取
#样值，因此采用short数据类型转换。现在我们得到的wave_data是一个一维的short类型的数组，但是因为我们的声音文件是双声
#道的，因此它由左右两个声道的取样交替构成：LRLRLRLR....LR（L表示左声道的取样值，R表示右声道取样值）。修改wave_data
#的sharp之后：
wave_data.shape = -1, 2
print(wave_data.shape)

#将其转置得到：
wave_data = wave_data.T
'''
for i in range(100):
    print(wave_data[0][44100 * 10 + i])
'''
'''
#最后通过取样点数和取样频率计算出每个取样的时间：
time = np.arange(0, nframes) * (1.0 / framerate)

# 绘制波形
pl.subplot(211) 
pl.plot(time, wave_data[0])
pl.subplot(212) 
pl.plot(time, wave_data[1], c="g")
pl.xlabel("time (seconds)")
pl.show()
'''
# 打开WAV文档
f = wave.open(r"partMusic.wav", "wb")
# 配置声道数、量化位数和取样频率
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(framerate)
# 将wav_data转换为二进制数据写入文件
f.writeframes(wave_data[0][:framerate * 10].tostring())
f.close()
