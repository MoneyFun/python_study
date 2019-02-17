#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 16:54:06 2019

@author: luke
"""

import cv2

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    # Capture frame-by-frame  
    _, frame = cap.read()  
    # Display the resulting frame  
    cv2.imshow('frame',frame)  
    key = cv2.waitKey(1)
    if key == ord('q'):  
        break
    elif key == ord('p'):
        cv2.imwrite("camera.jpg", frame)
# When everything done, release the capture  
cap.release()  
cv2.destroyAllWindows() 
