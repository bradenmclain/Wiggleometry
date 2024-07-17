import cv2
import time
from icecream import ic
from multiprocessing import Process, Queue
import schedule
import os
import numpy as np


cap = cv2.VideoCapture('stable_trim.mp4')
cap2 = cv2.VideoCapture('stability_deposit_1.mp4')

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 25)  
cap2.set(cv2.CAP_PROP_FPS, 25)  

state, frame = cap.read()
state2,frame2 = cap2.read()

while state or state2:
    state, frame = cap.read()
    state2 , frame2 = cap2.read()
    #cv2.imshow("preview", frame)
    cv2.imshow("preview2",frame2)
    cv2.waitKey(1)
