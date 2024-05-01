import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

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
        
    def find_countours(self,img,og_frame):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        marked_frame = cv2.drawContours(og_frame, contours, -1, (0,255,0), 3)
        print('finding contours')

        return marked_frame
    
    def find_ROI(thresh_image):
        pass
    
output = cv2.VideoWriter('wiggleometer.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 25, (1920,1080)) 
    
if __name__ == '__main__':
    wig = Wiggleometer()
    video = wig.importVideo('stable_trim.mp4',display=False)
    font = cv2.FONT_HERSHEY_SIMPLEX

    state, frame = video.read()
    thresh_prev = wig.threshold(frame)
    stability_values = []
    i = 0

    while state:
        thresh = wig.threshold(frame)
        dif = thresh - thresh_prev
        dif_number = np.sum(dif)
        display_img = wig.find_countours(thresh, frame)
        display_img = cv2.putText(display_img,f'Stability: {dif_number}',(10,70), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        if dif_number > 1000000:
            quality = 'Unstable'
            color = (0, 0, 255)
        else:
            quality = 'stable'
            color = (0, 255, 0)

        display_img = cv2.putText(display_img,quality,(10,140), font, 2, color, 2, cv2.LINE_AA)
        thresh_prev = thresh
        wig.display(display_img)
        state, frame = video.read()







    while state:
        thresh = wig.threshold(frame)
        dif = thresh - thresh_prev
        countoured = wig.find_countours(dif)
        dif_number = np.sum(dif)
        frame = wig.find_countours(thresh, frame)
        display_img = cv2.putText(frame,f'Stability: {dif_number}',(10,70), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        if dif_number > 1000000:
            quality = 'Unstable'
            color = (0, 0, 255)
        else:
            quality = 'stable'
            color = (0, 255, 0)
        display_img = cv2.putText(frame,quality,(10,140), font, 2, color, 2, cv2.LINE_AA)
        output.write(display_img)
        stability_values.append(dif_number)
        wig.display(display_img)
        thresh_prev = thresh
        state, frame = video.read()

    #plt.plot(stability_values)
    #plt.show()

    video = wig.importVideo('stubbing_trim.mp4',display=False)
    font = cv2.FONT_HERSHEY_SIMPLEX

    state, frame = video.read()
    thresh_prev = wig.threshold(frame)
    stability_values2 = []
    i = 0
    while state:
        thresh = wig.threshold(frame)
        dif = thresh - thresh_prev
        countoured = wig.find_countours(dif)
        dif_number = np.sum(dif)
        display_img = cv2.putText(frame,f'Stability: {dif_number}',(10,70), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        if dif_number > 1000000:
            quality = 'Unstable'
            color = (0, 0, 255)
        else:
            quality = 'stable'
            color = (0, 255, 0)
        display_img = cv2.putText(frame,quality,(10,140), font, 2, color, 2, cv2.LINE_AA)
        output.write(display_img)
        stability_values2.append(dif_number)
        #wig.display(frame)
        thresh_prev = thresh
        state, frame = video.read()


    video = wig.importVideo('balling_trim.mp4',display=False)
    font = cv2.FONT_HERSHEY_SIMPLEX

    state, frame = video.read()
    thresh_prev = wig.threshold(frame)
    stability_values3 = []
    i = 0
    while state:
        thresh = wig.threshold(frame)
        dif = thresh - thresh_prev
        countoured = wig.find_countours(dif)
        dif_number = np.sum(dif)
        display_img = cv2.putText(frame,f'Stability: {dif_number}',(10,70), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        if dif_number > 1000000:
            quality = 'Unstable'
            color = (0, 0, 255)
        else:
            quality = 'stable'
            color = (0, 255, 0)
        display_img = cv2.putText(frame,quality,(10,140), font, 2, color, 2, cv2.LINE_AA)
        output.write(display_img)
        stability_values3.append(dif_number)
        wig.display(frame)
        thresh_prev = thresh
        state, frame = video.read()

    output.release()
    video.release()
    plt.plot(stability_values)
    plt.plot(stability_values2)
    plt.plot(stability_values3)
    plt.show()

    