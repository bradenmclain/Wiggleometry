import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import collections
from scipy.stats import gmean
from draggable_lines import draggable_lines

class Wiggleometer: 

    def __init__(self, path,threshold):
        self.start_time = time.time()
        self.video = cv2.VideoCapture(path)
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.state, self.frame = self.video.read()
        self.pixel_count_buffer = collections.deque(maxlen = 30)
        self.binary_image_ring_buffer = collections.deque(maxlen = 2)
        self.gray_image_ring_buffer = collections.deque(maxlen = 2)
        self.frame_change_buffer = collections.deque(maxlen = 3)
        self.gray_change_buffer = collections.deque(maxlen = 3)
        self.white_pixel_buffer = collections.deque(maxlen = 3)
        self.red_buffer = collections.deque(maxlen = 3)
        self.frame_change_buffer.append(1)
        self.stability_state = "Not Depositing"
        self.height,self.width,val = self.frame.shape
        self.total_pix = self.height*self.width
        self.rgb_pix = []
        self.deposit_state = 'Initalize'
        self.threshold = threshold


        #thresholding parameters
        self.deposit_state_threshold = 3687626

        

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
        #threshold = 100
        #pick red channel
        self.red = self.frame[:, :, 2]
        self.red_buffer.append(np.sum(self.red))

        self.gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) 
        self.gray_image = cv2.GaussianBlur(src=self.gray_image, ksize=(5, 5), sigmaX=0.5)

        self.green_blue = self.frame[:,:,0]/2 + self.frame[:,:,1]/2

        self.binary_image = np.zeros_like(self.gray_image)
        self.binary_image[np.where(self.gray_image>self.threshold)] = 255

        # self.binary_image = np.zeros_like(self.red)
        # self.binary_image[np.where(self.red>self.threshold)] = 255

        #not currently using this
        # self.white = np.zeros_like(self.gray)
        # self.white[np.where(self.gray>230)] = 255
        # self.white_pixel_buffer.append(np.sum(self.white))


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
    
    def classification_analysis(self):
        self.gray_threshold()
        self.pixel_count_buffer.append(self.pixel_count)
        self.binary_image_ring_buffer.append(self.binary_image)
        self.gray_image_ring_buffer.append(self.gray_image)
        self.frame_change = np.sum(self.binary_image - self.binary_image_ring_buffer[0],dtype=np.float64)
        self.frame_change_buffer.append(self.frame_change)
        self.frame_change_difference = np.mean(np.asarray(self.frame_change_buffer,dtype=np.float64))
        self.gray_change = np.sum(self.gray_image - self.gray_image_ring_buffer[0],dtype=np.float64)
        self.gray_change_buffer.append(self.gray_change)
        self.gray_change_difference = np.mean(np.asarray(self.gray_change_buffer),dtype=np.float64)


    def get_stability_state(self):
        if self.deposit_state != 'Depositing':
            self.stability_state = 'Not Depositing'

        elif np.mean(test.red_buffer) > 25078982:
            self.stability_state = 'Balling'

        else:
            self.stability_state = 'Stable or Stubbing'

    
    def get_deposit_state(self):
        #find trim state

        if self.deposit_state == 'Initalize' and self.frame_change_difference > self.deposit_state_threshold:
            self.deposit_state = 'Trim'
        
        if self.deposit_state == 'Trim' and self.frame_change < 1000:
            self.deposit_state = 'Awaiting Deposition'

        if self.deposit_state == 'Awaiting Deposition' and self.frame_change_difference > self.deposit_state_threshold:
            self.deposit_state = 'Engage'

        if self.deposit_state == 'Engage' and self.frame_change_difference < self.deposit_state_threshold:
            self.deposit_state = 'Depositing'

        if self.deposit_state == 'Depositing' and self.frame_change_difference > self.deposit_state_threshold and self.stability_state != 'Balling':
            self.deposit_state = 'Retract'

        if self.deposit_state == 'Retract' and self.frame_change_difference < 1000:
            self.deposit_state = 'Deposition Complete'
        

        
        pass

        
   

    
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



if __name__ == '__main__':

    #test = Wiggleometer("./data/Second/wiggleometer_deposit_2.mp4")
    font = cv2.FONT_HERSHEY_SIMPLEX
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    #fourcc = -1
    
    i = 0
    global_binary_change = []
    global_total_red_pix = []
    global_gray_change = []
    videos = [1,3]
    thresholds = [10,40,60,80,100]
    for i,threshold  in enumerate(thresholds):
        
        
        file = f"./data/Second/wiggleometer_deposit_3.mp4"
        #file = ['stubbing_trim.mp4','stable_trim.mp4']
        print(f'the file is {file}')
        test = Wiggleometer(file,threshold)
        print(threshold)
        test.get_frame()
        height,width,val = test.frame.shape
        total_red_pix = []
        binary_change=[]
        white_count = []
        white_count_buffer = []
        gray_change = []
        plt.ion()

        while test.state:
            test.classification_analysis()
            test.get_deposit_state()
            test.get_stability_state()
            #print(test.deposit_state)
            display_img = cv2.putText(test.frame,test.stability_state,(10,140), font, 2, (255,255,255), 2, cv2.LINE_AA)
            display_img = cv2.putText(display_img,test.deposit_state,(10,200), font, 2, (255,255,255), 2, cv2.LINE_AA)
           
            left_box_vals = test.gray_image[:,0:300]
            right_box_vals = test.gray_image[:,1620:1920]
            box_vals = np.append(left_box_vals,right_box_vals)
            hist = np.histogram(box_vals, bins=255, range=[0, 255])[0]
            print(hist)
            thresh = np.max(box_vals)
            print(f'max "black" value is :{thresh}')
            # print(test.gray_image)
            # print('made it')
            plt.plot(hist)
            plt.xlim([5,70])
            plt.ylim([0,100])
            
            # plt.title("Histogram with 'auto' bins")
            plt.draw()
            plt.pause(.01)
            plt.clf()
            #out.write(display_img)
            cv2.imshow('frame',test.gray_image)
            if thresh > 10:
                cv2.waitKey(2000)
            else:
                cv2.waitKey(10)
            test.get_frame()
            #total_red_pix.append(np.sum(test.red))
            total_red_pix.append(np.mean(test.red_buffer))
            #frame_change.append(np.average(test.frame_change_difference))
            binary_change.append(np.mean(np.asarray(test.frame_change_buffer,dtype=np.float64)))
            gray_change.append(np.mean(np.asarray(test.gray_change_buffer,dtype=np.float64)))
            #white_count_buffer.append(np.average(test.white_pixel_buffer))
            #white_count.append(np.sum(test.white))
            # plt.plot(frame_change)
            # plt.draw()
            # plt.pause(.01)
            # plt.clf()

        #plt.ioff()
        #global_white_count.append(white_count)
        #global_white_count_buffer.append(white_count_buffer)
        global_total_red_pix.append(total_red_pix)
        global_binary_change.append(binary_change)
        global_gray_change.append(gray_change)

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
    titles = ['Stable-Oscillating Deposition 1', 'Dripping Deposition','Stable Deposition 1', 'Stable-Oscillating Deposition 2', 'Oscillating Deposition', 'Stable-Oscillating Deposition 3', 'Stable Deposition 2']

    for idx,threshold in enumerate(thresholds):
    #     #plt.plot(vid)
        plt.plot(global_total_red_pix[idx],label = f'{threshold}')
        #plt.plot(global_white_count_buffer[idx],label = f'deposit video {idx}')
    plt.legend(loc="upper left")
    plt.show()


    for idx,threshold in enumerate(thresholds):
    #     #plt.plot(vid)
        plt.plot(global_gray_change[idx],label = f'{threshold}')
        #plt.plot(global_white_count_buffer[idx],label = f'deposit video {idx}')
    plt.legend(loc="upper left")

    plt.xlabel('Frame')
    plt.ylabel('Frame to Frame Pixel Change')
    plt.title('Frame to Frame Gray Pixel Change for Stable and Oscillating Deposition')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplot(111)
    plt.legend(loc='upper right')

    for idx,threshold in enumerate(thresholds):
    #     #plt.plot(vid)
        plt.plot(global_binary_change[idx],label = f'{threshold}')
        #plt.plot(global_white_count_buffer[idx],label = f'deposit video {idx}')
    plt.legend(loc="upper left")

    plt.xlabel('Frame')
    plt.ylabel('Frame to Frame Pixel Change')
    plt.title('Frame to Frame Pixel Change for Stable and Oscillating Deposition')

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

