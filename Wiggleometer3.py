import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import collections

class Wiggleometer: 

    def __init__(self, path):
        self.start_time = time.time()
        self.video = cv2.VideoCapture(path)
        self.state, self.frame = self.video.read()
        self.binary_ring_buffer = collections.deque(maxlen = 10)
        self.binary_image_ring_buffer = collections.deque(maxlen = 2)
        

        self.pixel_count = 0


    def get_frame(self):
        self.state, self.frame = self.video.read()
        return 
    
    def threshold_image(self,**kwargs):
        colors = {
            'red' : 255,
            'green' : 255,
            'blue' : 255
        }
        
        for key, value in kwargs.items():
            colors.update({key:value})

        red = self.frame[:, :, 2]
        green = self.frame[:,:,1]
        blue = self.frame[:,:,0]
        self.color_binary_image = np.zeros_like(red)
        self.color_binary_image[np.where(red>colors['red'])] = 255
        self.color_binary_image[np.where(green>colors['green'])] = 255
        self.color_binary_image[np.where(blue>colors['blue'])] = 255
        self.pixel_count = np.sum(self.color_binary_image)
        self.binary_image_ring_buffer.append(self.color_binary_image)

        return 
    
    def gray_threshold(self,threshold):
        gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) 
        self.binary_image = np.zeros_like(gray_image)
        self.binary_image[np.where(gray_image>threshold)] = 255
        self.pixel_count = np.sum(self.binary_image)

    def binary_pixel_count_look_back(self):
        self.binary_ring_buffer.append(self.pixel_count)
        return np.mean(self.binary_ring_buffer)


    def resize_frame(self):
        self.frame = cv2.resize(self.frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
    
    def find_countours(self):
        contours, hierarchy = cv2.findContours(self.binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.frame_with_contours = cv2.drawContours(self.frame, contours, -1, (0,255,0), 3)
        print('finding contours')

        return self.frame_with_contours
    
    def get_previous_frame(self):
        return self.image_ring_buffer[1]

    
    def binary_frame_change_count(self):
        return self.binary_image - self.binary_image_ring_buffer[0]
        
    
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
    

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == '__main__':


    stable = Wiggleometer("stable_trim.mp4")
    stubby = Wiggleometer("stubbing_trim.mp4")
    balling = Wiggleometer("balling_trim.mp4")
    stable.get_frame()
    stubby.get_frame()
    balling.get_frame()
    stable_pix = []
    stable_pix_lookback = []
    stubby_pix = []
    stubby_pix_lookback = []
    balling_pix = []
    balling_pix_lookback = []
    stable_change = []
    stubby_change = []
    balling_change = []
    i = 0


    big = True
    while balling.state or stubby.state or stable.state:
        i+=1

        if stable.state:
            stable.resize_frame()
            stable.threshold_image(red=200)
            stable.gray_threshold(252)
            stable_pix.append(stable.pixel_count)
            stable_pix_lookback.append(stubby.binary_pixel_count_look_back())
            stable_change.append(np.sum(stable.binary_frame_change_count()))
            stable.get_frame()

        if stubby.state:
            stubby.resize_frame()
            stubby.threshold_image(red=200)
            stubby.gray_threshold(252)
            stubby_pix.append(stubby.pixel_count)
            stubby_pix_lookback.append(stable.binary_pixel_count_look_back())
            stubby_change.append(np.sum(stubby.binary_frame_change_count()))
            stubby.get_frame()

        if balling.state:
            balling.resize_frame()
            balling.threshold_image(red=200)
            balling.gray_threshold(252)
            balling_pix.append(balling.pixel_count)
            balling_pix_lookback.append(balling.binary_pixel_count_look_back())
            balling_change.append(np.sum(balling.binary_frame_change_count()))
            balling.get_frame()
            # cv2.imshow('frame',balling.binary_image)
            # cv2.waitKey(5)
      
        #start = time.time()
        
        #circles = stable.cv_hough_circle()
        print(f'Stable is {stable.state} Stubbing is {stubby.state} Balling is {balling.state} Big is {big}', end='\r')
        # cv2.imshow('frame',stable.frame)
        # cv2.waitKey(5)
        
    stable_change = np.array(stable_change,dtype= np.float64)
    stubby_change = np.array(stubby_change,dtype= np.float64)
    print(stable_change[450:500])
    print(np.diff(stable_change[450:500]))

    plt.plot(stable_change,'g', label = 'Stable')
    plt.plot(stubby_change,'b',label = 'Stubby')
    #plt.plot(balling_change,'orange', label = 'Balling')
    plt.xlabel("Frame")
    plt.ylabel("Difference between Current and Previous Frame")
    plt.legend(loc='best')
    plt.show()
    plt.clf()


    # plt.plot(np.abs(np.diff(np.array(stable_change))),'g',label = 'Stable Difference')
    # plt.plot(np.abs(np.diff(np.array(stubby_change))),'b',label = 'Stubby Difference')
    plt.plot(moving_average(np.abs(np.diff(np.array(stable_change))),10),'g',label = 'Stable')
    plt.plot(moving_average(np.abs(np.diff(np.array(stubby_change))),10),'b',label = 'Stubby')
    plt.legend(loc='best')
    plt.xlabel("Frame")
    plt.ylabel("Rate of Change of Current Frame - Previous Frame")
    plt.show()
    plt.clf()

    plt.plot(stable_pix_lookback,'g', label = 'Stable')
    plt.plot(stubby_pix_lookback,'b',label = 'Stubby')
    plt.plot(balling_pix_lookback,'orange', label = 'Balling')
    plt.xlabel("Frame")
    plt.ylabel("Total Thresholded Pixel Count")
    plt.legend(loc='best')

    plt.show()

    plt.clf()

    plt.plot(stable_pix,'g')
    plt.plot(stubby_pix,'b')
    plt.plot(balling_pix,color='orange')
    plt.show()

    
            
            
        
    

