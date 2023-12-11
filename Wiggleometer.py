import cv2
import time
import numpy as np

class Wiggleometer: 

    def __init__(self):
        self.start_time = time.time()

    def makeBin(self,frame):
        blurred = cv2.GaussianBlur(src=frame, ksize=(5, 5), sigmaX=0.5)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        bin_image = np.zeros_like(gray) 
        bin_image[gray > np.min(gray)*1.01] = 255
        return bin_image
    
    def threshold(self,image):
        red = image[:, :, 2]
        empty = np.zeros_like(red)
        empty[np.where(red>150)] = 255
        return empty        


    def importVideo(self,dest,display=False):
        cap = cv2.VideoCapture(dest)
        state, frame = cap.read()
        if display:
            while state:
                cv2.imshow('color_frame',frame)
                cv2.waitKey(1)
                state, frame = cap.read()
        return cap
    
    def display(self,img):
        if img.size > 0:
            cv2.imshow('frame',img)
            cv2.waitKey(5)

    def cannyEdgeDetect(self,img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h = hsv[:, :, 1]
      
        # Apply Gaussian blur to reduce noise and smoothen edges 
        blurred = cv2.GaussianBlur(src=h, ksize=(5, 5), sigmaX=0.5) 
        
        # Perform Canny edge detection 
        edges = cv2.Canny(blurred, 150, 200) 

        return edges
        
    def find_countours(self,img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
        return img
    
    def find_ROI(thresh_image):
        pass

    
if __name__ == '__main__':
    wig = Wiggleometer()
    video = wig.importVideo('stubbing.mp4',display=False)

    state, frame = video.read()
    thresh_prev = wig.threshold(frame)
    i = 0
    while state:
        state, frame = video.read()
        thresh = wig.threshold(frame)
        dif = thresh - thresh_prev
        countoured = wig.find_countours(dif)
        wig.display(dif)
        thresh_prev = thresh
        


    