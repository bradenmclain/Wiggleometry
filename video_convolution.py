import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import convolve

kernel_circ = np.array([[-1,0,0,1,1,1,0,0,-1],
               [-1,0,1,1,-2,1,1,0,-1],
               [-1,1,1,-2,-2,-2,1,1,-1],
               [-1,1,-2,-2,-2,-2,-2,1,-1],
               [1,1,-2,-2,-2,-2,-2,1,1],
               [-1,1,-2,-2,-2,-2,-2,1,-1],
               [-1,1,1,-2,-2,-2,1,1,-1],
               [-1,1,1,1,-2,1,1,1,-1],
               [-1,-2,-3,1,1,1,-3,-2,-1]])

cap = cv2.VideoCapture('balling_trim.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    
    start = time.time()
    small_frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)
    cv_conv_img = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel_circ)
    cv_conv_img = cv_conv_img * 255 / cv_conv_img.max()
    cv_conv_img = cv_conv_img.astype(np.uint8)
    #end = time.time()

    print(f'it legit took {time.time() - start} seconds')
    cv2.imshow('frame',small_frame)
    cv2.imshow('conv_frame',cv_conv_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()