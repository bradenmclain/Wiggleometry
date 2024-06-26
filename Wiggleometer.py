import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import collections
from scipy.stats import gmean

class Wiggleometer: 

    def __init__(self, path):
        self.start_time = time.time()
        self.video = cv2.VideoCapture(path)
        self.state, self.frame = self.video.read()
        self.pixel_count_buffer = collections.deque(maxlen = 30)
        self.binary_image_ring_buffer = collections.deque(maxlen = 3)
        self.frame_change_buffer = collections.deque(maxlen = 15)
        self.frame_change_buffer.append(1)
        self.deposit_state = "Empty"
        self.height,self.width,val = self.frame.shape
        self.total_pix = self.height*self.width

        

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

        return 
    
    def gray_threshold(self):
        threshold = 252
        gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) 
        self.binary_image = np.zeros_like(gray_image)
        self.binary_image[np.where(gray_image>threshold)] = 1
        self.pixel_count = np.sum(self.binary_image)

    def binary_pixel_count_look_back(self):
        self.pixel_count_buffer.append(self.pixel_count)
        return np.mean(self.pixel_count_buffer)


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
    
    def get_state(self):
        #self.resize_frame
        self.gray_threshold()
        self.pixel_count_buffer.append(self.pixel_count)
        self.binary_image_ring_buffer.append(self.binary_image)
        self.frame_change_buffer.append(np.sum(self.binary_image - self.binary_image_ring_buffer[0],dtype=np.float64))
        self.frame_change_difference = np.abs(np.diff(np.asarray(self.frame_change_buffer,dtype=np.float64)))
        if np.mean(self.pixel_count_buffer) > .0085*self.total_pix:
            self.deposit_state = 'Balling'
        elif np.mean(self.pixel_count_buffer) < .0005*self.total_pix:
            self.deposit_state = 'Empty'
        elif np.mean(self.frame_change_difference) < 250000:
            self.deposit_state = 'Stable'
        else:
            self.deposit_state = 'Stubbing'

        
   

    
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



if __name__ == '__main__':

    test = Wiggleometer("combined.mp4")
    font = cv2.FONT_HERSHEY_SIMPLEX
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    #fourcc = -1
    test.get_frame()
    height,width,val = test.frame.shape
    out = cv2.VideoWriter('output.mkv', fourcc, 30.0, (width,height))


    stable = Wiggleometer("stable_trim.mp4")
    stubby = Wiggleometer("stubbing_trim.mp4")
    balling = Wiggleometer("balling_trim.mp4")
    height,width,val = stable.frame.shape
    total_pix = height * width
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
    gmean_stable = []
    gmean_stubby = []
    gmean_balling = []
    i = 0

    while test.state:
        test.get_state()
        deposit_state = test.deposit_state
        print(deposit_state)
        display_img = cv2.putText(test.frame,deposit_state,(10,140), font, 2, (255,255,255), 2, cv2.LINE_AA)
        out.write(display_img)
        cv2.imshow('frame',display_img)
        cv2.waitKey(5)
        test.get_frame()


    # big = True
    # out.release()
    
    # while balling.state or stubby.state or stable.state:
    #     i+=1

    #     if stable.state:
    #         stable.get_state()
    #         deposit_state = stable.deposit_state
    #         stable_pix.append(np.mean(stable.pixel_count_buffer))
    #         stable_change.append(np.mean(stable.frame_change_difference))
    #         stable.get_frame()

    #     if stubby.state:
    #         stubby.get_state()
    #         deposit_state = stubby.deposit_state
    #         stubby_pix.append(np.mean(stubby.pixel_count_buffer))
    #         stubby_change.append(np.mean(stubby.frame_change_difference))
    #         stubby.get_frame()

    #     if balling.state:
    #         balling.get_state()
    #         deposit_state = balling.deposit_state
    #         balling_pix.append(np.mean(balling.pixel_count_buffer))
    #         balling_change.append(np.mean(balling.frame_change_difference))
    #         balling.get_frame()
            # cv2.imshow('frame',balling.binary_image)
            # cv2.waitKey(5)
      
        #start = time.time()
        
        #circles = stable.cv_hough_circle()
        #print(f'Stable is {stable.state} Stubbing is {stubby.state} Balling is {balling.state} Big is {big}', end='\r')
        # cv2.imshow('frame',stable.frame)
        # cv2.waitKey(5)

    plt.plot(600,stubby.height*stubby.width*.0085,'*')
    plt.plot(stable_pix,'g', label = 'Stable')
    plt.plot(stubby_pix,'b',label = 'Stubby')
    plt.plot(balling_pix,'orange', label = 'Balling')
    plt.xlabel("Frame")
    plt.ylabel("Total Thresholded Pixel Count")
    plt.legend(loc='best')
    plt.show()
    plt.clf()
    # plt.show()
        
    # # stable_change = np.array(stable_change,dtype= np.float64)
    # # stubby_change = np.array(stubby_change,dtype= np.float64)


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
    # plt.plot(moving_average(np.abs(np.diff(np.array(stable_change))),10),'g',label = 'Stable')
    # plt.plot(moving_average(np.abs(np.diff(np.array(stubby_change))),10),'b',label = 'Stubby')
    # plt.legend(loc='best')
    # plt.xlabel("Frame")
    # plt.ylabel("Rate of Change of Current Frame - Previous Frame")
    # plt.show()
    # plt.clf()

    
            
            
        
    

