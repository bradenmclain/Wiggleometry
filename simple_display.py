import cv2
import time
from icecream import ic
from multiprocessing import Process, Queue
import schedule
import os
import numpy as np


cap = cv2.VideoCapture(0,cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)  

state, frame = cap.read()

while state:
    state, frame = cap.read()
    cv2.imshow("preview", frame)
    cv2.waitKey(1)
