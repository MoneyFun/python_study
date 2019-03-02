# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
url = 'http://admin:admin@192.168.43.1:8081'
cap = cv2.VideoCapture(url)
while(cap.isOpened()):  
    # Capture frame-by-frame  
    ret, frame = cap.read()  
    # Display the resulting frame  
    cv2.imshow('frame',frame)  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
# When everything done, release the capture  
cap.release()  
cv2.destroyAllWindows()  