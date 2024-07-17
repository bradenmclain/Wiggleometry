import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import collections
from scipy.stats import gmean
from draggable_lines import draggable_lines

class Wiggleometer: 

    def __init__(self, path):
        self.start_time = time.time()
        self.video = cv2.VideoCapture(path)
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.state, self.frame = self.video.read()
        self.pixel_count_buffer = collections.deque(maxlen = 30)
        self.binary_image_ring_buffer = collections.deque(maxlen = 2)
        self.gray_image_ring_buffer = collections.deque(maxlen = 2)
        self.frame_change_buffer = collections.deque(maxlen = 3)
        self.white_pixel_buffer = collections.deque(maxlen = 3)
        self.frame_change_buffer.append(1)
        self.deposit_state = "Empty"
        self.height,self.width,val = self.frame.shape
        self.total_pix = self.height*self.width
        self.rgb_pix = []

        

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
        self.white = np.zeros_like(red)
        self.white[np.where(self.red>250)] = 255

        return 
    
    def gray_threshold(self):
        threshold = 100
        self.red = self.frame[:, :, 2]
        self.green = self.frame[:,:,1]
        self.blue = self.frame[:,:,0]
        self.gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) 
        self.gray = cv2.GaussianBlur(src=self.gray_image, ksize=(5, 5), sigmaX=0.5)
        #test.frame = gray_image
        self.binary_image = np.zeros_like(self.gray_image)
        self.binary_image[np.where(self.gray_image>threshold)] = 255
        self.white = np.zeros_like(self.gray)
        self.white[np.where(self.gray>230)] = 255
        self.white_pixel_buffer.append(np.sum(self.white))


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
        self.gray_image_ring_buffer.append(self.gray_image)
        self.frame_change_buffer.append(np.sum(self.binary_image - self.binary_image_ring_buffer[0],dtype=np.float64))
        #self.frame_change_difference = np.abs(np.diff(np.asarray(self.frame_change_buffer,dtype=np.float64)))
        self.frame_change_difference = np.mean(np.asarray(self.frame_change_buffer,dtype=np.float64))
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

    test = Wiggleometer("./data/Second/wiggleometer_deposit_2.mp4")
    font = cv2.FONT_HERSHEY_SIMPLEX
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    #fourcc = -1
    test.get_frame()
    height,width,val = test.frame.shape
    
    i = 0
    global_frame_change = []
    global_total_red_pix = []
    global_white_count = []
    global_white_count_buffer = []
    for vid,i  in enumerate([2,3,5]):
        file = f"./data/Second/wiggleometer_deposit_{i}.mp4"
        #file = ['stubbing_trim.mp4','stable_trim.mp4']
        print(f'the file is {file}')
        test = Wiggleometer(file)
        test.get_frame()
        total_red_pix = []
        frame_change=[]
        white_count = []
        white_count_buffer = []
        #plt.ion()

        while test.state:
            test.get_state()
            deposit_state = test.deposit_state
            display_img = cv2.putText(test.frame,deposit_state,(10,140), font, 2, (255,255,255), 2, cv2.LINE_AA)
            #out.write(display_img)
            # cv2.imshow('frame',test.frame)
            # cv2.waitKey(1)
            test.get_frame()
            total_red_pix.append(np.sum(test.red))
            frame_change.append(np.average(test.frame_change_difference))
            white_count_buffer.append(np.average(test.white_pixel_buffer))
            white_count.append(np.sum(test.white))
            # plt.plot(frame_change)
            # plt.draw()
            # plt.pause(.01)
            # plt.clf()

        #plt.ioff()
        global_white_count.append(white_count)
        global_white_count_buffer.append(white_count_buffer)
        global_total_red_pix.append(total_red_pix)
        global_frame_change.append(frame_change)

        #plt.plot(total_red_pix)
        #plt.plot(frame_change)
        #plt.plot(np.abs(np.gradient(frame_change)))
        # plt.plot(frame_change,color = 'blue')
        # plt.show()
        # plt.plot(frame_change,color = 'blue')
        # plt.plot(((np.gradient(frame_change))),color = 'green')
        # plt.show()
        # plt.plot(frame_change,color = 'blue')
        # plt.plot((np.gradient(np.gradient(frame_change))),color = 'red')

        # plt.show()
        # plt.clf()
    titles = ['Dripping Deposition','Stable Deposition','Oscillating Deposition']

    for idx,vid in enumerate(global_total_red_pix):
    #     #plt.plot(vid)
        plt.plot(global_white_count[idx],label = titles[idx])
        #plt.plot(global_white_count_buffer[idx],label = f'deposit video {idx}')
    plt.legend(loc="upper left")
    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplot(111)
    plt.legend(loc='upper right')

    for idx,vid in enumerate(global_total_red_pix):
    #     #plt.plot(vid)
        plt.plot(global_total_red_pix[idx],label = titles[idx])
        #plt.plot(global_white_count_buffer[idx],label = f'deposit video {idx}')
    plt.legend(loc="upper left")

    plt.xlabel('Frame')
    plt.ylabel('Total Pixel Intensity')
    plt.title('Total Pixel Intensity for Various Deposition States')

    Vline = draggable_lines(ax, "h", 10000,len(max(global_total_red_pix, key=len)))
    # Update the legend after adding the draggable line
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')
        
        # plt.plot(pixel_count_array)
        # plt.show()
    while Vline.XorY > 0:
        handles, labels = ax.get_legend_handles_labels()
        print(f'the position is {Vline.XorY}')
        plt.draw()
        plt.pause(.1)

        plt.show()

