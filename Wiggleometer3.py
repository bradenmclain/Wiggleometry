import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

class Wiggleometer: 

    def __init__(self, path):
        self.start_time = time.time()
        self.video = cv2.VideoCapture(path)
        self.state, self.frame = self.video.read()

    def get_frame(self):
        self.state, self.frame = self.video.read()
        return 
    
    def threshold_image(self,value):
        red = self.frame[:, :, 2]
        self.binary_image = np.zeros_like(red)
        self.binary_image[np.where(red>value)] = 255
        self.pixel_count = np.sum(self.binary_image)
        return 
    
    def resize_frame(self):
        self.frame = cv2.resize(self.frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
    
    def find_countours(self):
        contours, hierarchy = cv2.findContours(self.binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.frame_with_contours = cv2.drawContours(self.frame, contours, -1, (0,255,0), 3)
        print('finding contours')

        return self.frame_with_contours
    
    def cv_hough_circle(self):
        blurred = cv2.GaussianBlur(src=self.frame, ksize=(5, 5), sigmaX=0.5)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(self.binary_image, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=30, maxRadius=100)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(gray, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(gray, center, radius, (255, 0, 255), 3)

        return gray
    
    

if __name__ == '__main__':
    stable = Wiggleometer("stable_trim.mp4")
    stubby = Wiggleometer("stubbing_trim.mp4")
    balling = Wiggleometer("balling_trim.mp4")
    stable.get_frame()
    stubby.get_frame()
    balling.get_frame()
    stable_pix = []
    stubby_pix = []
    balling_pix = []
    i = 0

    big = True
    while stable.state:
        i+=1

        if stable.state:
            stable.resize_frame()
            stable.threshold_image(150)
            stable_pix.append(stable.pixel_count)

        if stubby.state:
            stubby.resize_frame()
            stubby.threshold_image(150)
            stubby_pix.append(stubby.pixel_count)

        if balling.state:
            balling.resize_frame()
            balling.threshold_image(150)
            balling_pix.append(balling.pixel_count)
      
        #start = time.time()
        
        #circles = stable.cv_hough_circle()
        print(f'Stable is {stable.state} Stubbing is {stubby.state} Balling is {balling.state} Big is {big}', end='\r')
        # cv2.imshow('frame',stable.frame)
        # cv2.waitKey(5)
        stable.get_frame()
        stubby.get_frame()
        balling.get_frame()
        
        if stable.state == False:
            if balling.state == False:
                if stubby.state == False:
                    big == False
        

    plt.plot(stable_pix)
    plt.plot(stubby_pix)
    plt.plot(balling_pix)
    plt.show()

    
            
            
        
    

