import numpy as np
import cv2

cap = cv2.VideoCapture("stability_deposit_1.h264")


font = cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc(*'H264')



out = cv2.VideoWriter('output1.mkv', fourcc, 30.0, (1920,1080))

import cv2 
 
# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture('Resources/Cars.mp4')
 
if (cap.isOpened() == False):
  print("Error opening the video file")
# Read fps and frame count

 
while(cap.isOpened()):
  # vid_capture.read() methods returns a tuple, first element is a bool 
  # and the second is frame
  ret, frame = cap.read()
  if ret == True:
    cv2.imshow('Frame',frame)
    # 20 is in milliseconds, try to increase the value, say 50 and observe
    key = cv2.waitKey(20)
     
    if key == ord('q'):
      break
  else:
    break
 
# Release the video capture object
cap.release()