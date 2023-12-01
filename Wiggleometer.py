import cv2
import time
import numpy as np

class Wiggleometer: 

    def __init__(self):
        self.start_time = time.time()

    def makeBin(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bin_image = np.zeros_like(gray) 
        bin_image[gray > np.min(gray)*1.01] = 255
        print(bin_image.size)
        return bin_image
    
    def importVideo(self,dest,display=False):
        cap = cv2.VideoCapture(dest)
        state, frame = cap.read()
        if display:
            while state:
                bin_image = self.makeBin(frame)
                # Check if bin_image is not empty before displaying
                if bin_image.size > 0:
                    cv2.imshow('frame', bin_image)
                    print(np.size(bin_image))
                else:
                    print("Empty bin_image, skipping display")      
                              
                cv2.imshow('frame',bin_image)
                cv2.imshow('color_frame',frame)
                cv2.waitKey(1)
                state, frame = cap.read()
                print(state)
        return cap
    


    
if __name__ == '__main__':
    wig = Wiggleometer()
    video = wig.importVideo('Test7.mp4',display=True)

    