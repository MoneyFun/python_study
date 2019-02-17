#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 18:03:15 2019

@author: luke
"""

import cv2

#获得视频的格式
videoCapture = cv2.VideoCapture('fengkuangdewaixingren.ts')

#获得码率及尺寸
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
 
 
#读帧
success, frame = videoCapture.read()
while success :
    cv2.imshow('windows', frame) #显示
    cv2.waitKey(1000/int(fps)) #延迟
    success, frame = videoCapture.read() #获取下一帧
 
videoCapture.release()
