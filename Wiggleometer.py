import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import collections
from scipy.stats import gmean
from draggable_lines import draggable_lines
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import gaussian_filter1d

class Wiggleometer: 

    def __init__(self, path,threshold):
        self.start_time = time.time()
        self.video = cv2.VideoCapture(path)
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.state, self.frame = self.video.read()
        self.total_intensity_buffer = collections.deque(maxlen = 3)
        self.binary_image_ring_buffer = collections.deque(maxlen = 2)
        self.frame_change_buffer = collections.deque(maxlen = 3)
        self.binary_pixel_count_buffer = collections.deque(maxlen = 3)
        self.frame_change_buffer.append(1)
        self.stability_state = "Not Depositing"
        self.height,self.width,val = self.frame.shape
        self.total_pix = self.height*self.width
        self.rgb_pix = []
        self.deposit_state = 'Initalize'
        self.threshold = threshold
        self.stub_count = 0
        self.stub_frequency_buffer = collections.deque(maxlen = 10)
        self.sec_der_frequency_buffer = collections.deque(maxlen = 10)
        self.stub_indecies = []
        self.frame_idx = 0
        self.pixel_count = 0
        self.trim_index = []
        self.engage_index = []
        self.retract_index = []
        


        #thresholding parameters
        self.deposit_state_threshold = 3687626
        self.balling_threshold = 20248446
        self.stubbing_threshold = 801825
        


    def get_frame(self):
        self.state, self.frame = self.video.read()
        self.frame_idx+= 1
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
        self.no_red = round(((green + blue)/np.max(green+blue))*255)
        self.color_binary_image = np.zeros_like(red)
        self.color_binary_image[np.where(red>colors['red'])] = 255
        self.color_binary_image[np.where(green>colors['green'])] = 255
        self.color_binary_image[np.where(blue>colors['blue'])] = 255
        self.white = np.zeros_like(red)
        self.white[np.where(self.red>250)] = 255

        return 
    
    def gray_threshold(self):
        
        self.gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) 
        self.gray_image = cv2.GaussianBlur(src=self.gray_image, ksize=(5, 5), sigmaX=0.5)


        self.binary_image = np.zeros_like(self.gray_image)
        self.binary_image[np.where(self.gray_image>self.threshold)] = 255

        binary_pixel_count = np.sum(self.binary_image)
        self.binary_pixel_count_buffer.append(binary_pixel_count)

    def resize_frame(self):
        self.frame = cv2.resize(self.frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
    
    def find_countours(self):
        contours, hierarchy = cv2.findContours(self.binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.frame_with_contours = cv2.drawContours(self.frame, contours, -1, (0,255,0), 3)
        print('finding contours')

        return self.frame_with_contours
    
    def get_previous_frame(self):
        return self.image_ring_buffer[1]
    
    def adjust_binary_treshold(self):
        pass
    
    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
        
    
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
        self.binary_image_ring_buffer.append(self.binary_image)
        self.total_intensity = np.sum(self.gray_image)
        self.total_intensity_buffer.append(self.total_intensity)
        self.frame_change = np.sum(self.binary_image - self.binary_image_ring_buffer[0],dtype=np.float64)
        self.frame_change_buffer.append(self.frame_change)
        self.frame_change_difference = np.mean(np.asarray(self.frame_change_buffer,dtype=np.float64))
        self.stub_frequency_buffer.append(self.frame_change)

    def get_stability_state(self):

        #print(self.stub_count)
        if self.deposit_state != 'Depositing':
            self.stability_state = 'Not Depositing'

        elif np.mean(self.total_intensity_buffer) > self.balling_threshold:
            self.stability_state = 'Balling'
            print('it was balling maybe?')

        else:
            start = time.time()
            peaks,__ = find_peaks(np.asarray(self.stub_frequency_buffer),prominence = 2000000,plateau_size = 1,width=1)
            end = time.time()- start
            #print(f'it took {end} seconds for an 8 sized one')
            prominences = peak_prominences(np.asarray(self.stub_frequency_buffer), peaks)[0]
            print(peaks)
            print(prominences)
            if np.mean(np.asarray(self.frame_change_buffer,dtype=np.float64)) > self.stubbing_threshold:
                self.stability_state = 'Stubbing'
                self.stub_count +=1 
            else:
                self.stability_state = 'Stable' 
                #print('i was stable')
            if len(peaks) != 0:
                
                self.stub_indecies.append(self.frame_idx-(10-peaks[0]))
                self.stability_state = 'Stubbing'

            if len(self.stub_indecies) == 1 and self.stub_indecies[0] + 12 >= self.frame_idx:
                self.stability_state = 'Stubbing'

            if len(self.stub_indecies) > 1 and ((self.stub_indecies[-1]) - self.stub_indecies[-2])*2 + self.stub_indecies[-1] > self.frame_idx:
                self.stability_state = 'Stubbing'                


    def adjust_threshold(self):
        pass
    
    
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
            print('Deposit has started')

        if self.deposit_state == 'Depositing' and self.frame_change_difference > self.deposit_state_threshold and self.stability_state != 'Balling':
            self.deposit_state = 'Retract'

        if self.deposit_state == 'Retract' and self.frame_change_difference < 1000:
            self.deposit_state = 'Deposition Complete'


        #record when things happen
        if self.deposit_state == 'Trim':
            self.trim_index.append(self.frame_idx)
        
        if self.deposit_state == 'Engage':
            self.engage_index.append(self.frame_idx)

        if self.deposit_state == 'Retract':
            self.retract_index.append(self.frame_idx)


        
    def get_median_pixel_intensity(self):
        bound_box = test.gray_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

                
        hist_box, bins = np.histogram(bound_box, bins=255, range=[0, 255])

        cumulative_dist = np.cumsum(hist_box[threshold:])

        # Calculate the total number of pixels
        total_pixels_above_threshold = cumulative_dist[-1]


        # Find the median pixel value
        median_value = np.searchsorted(cumulative_dist, total_pixels_above_threshold / 2) + 100

        return median_value

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



if __name__ == '__main__':

    #test = Wiggleometer("./data/Second/wiggleometer_deposit_2.mp4")
    font = cv2.FONT_HERSHEY_SIMPLEX
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    #fourcc = -1
    
    i = 0
    global_binary_change = []
    global_median = []
    global_total_pix = []
    global_total_intensity = []
    videos = [1,2,3,4,5,6,7]
    files = [1,2,3,4,5,6,7]
    roi = [795,444,305,588]
    threshold = 100

    for i,file  in enumerate(files):
        file = f"./data/Second/wiggleometer_deposit_{file}.mp4"
        print(f'the file is {file}')
        test = Wiggleometer(file,threshold)
        
        test.get_frame()
        height,width,val = test.frame.shape
        total_red_pix = []
        binary_change=[]
        white_count = []
        white_count_buffer = []
        gray_change = []
        median = []
        total_intensity = []
        total_pix = []
        frame = 0
        peak_indexs = []

        stub_idx = 0
        

        while test.state:
            start = time.time()
            test.classification_analysis()
            test.get_deposit_state()
            test.get_stability_state()
            #test.get_stability_state()
            display_img = cv2.putText(test.frame,test.stability_state,(10,140), font, 2, (255,255,255), 2, cv2.LINE_AA)
            display_img = cv2.putText(display_img,test.deposit_state,(10,200), font, 2, (255,255,255), 2, cv2.LINE_AA)
            #contours, _ = cv2.findContours(test.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find the convex hull of the largest contour
            # if contours:
            #     largest_contour = max(contours, key=cv2.contourArea)
            #     hull = cv2.convexHull(largest_contour)

            #     # Draw the convex hull on a new image
            #     hull_image = np.zeros_like(test.binary_image)
            #     cv2.drawContours(hull_image, [hull], 0, 255, thickness=cv2.FILLED)
            #     wire = hull_image - test.binary_image
                # cv2.imshow('hull image',hull_image)
                # cv2.imshow('wire',wire)
                # cv2.waitKey(10)
            
           
            #if test.deposit_state == 'Depositing':
            #     start = time.time()
            #     median.append(test.get_median_pixel_intensity())
                
            #     #total_intensity.append(np.sum(test.gray_image))
            #     print(f'that took {time.time()-start} seconds')
            
            binary_change.append(np.mean(np.asarray(test.frame_change_buffer,dtype=np.float64)))
            total_pix.append(np.sum(test.binary_image))
            cv2.rectangle(test.gray_image, (int(roi[0]), int(roi[1])), (int(roi[0]+roi[2]), int(roi[1]+roi[3])), (255, 255, 255), 2) 
            # cv2.imshow('frame',display_img)
            # cv2.waitKey(10)

            test.get_frame()
            
            total_intensity.append(np.mean(np.asarray(test.total_intensity_buffer,dtype=np.float64)))
            

        #print(peak_indexs)

        
        der = np.gradient((binary_change))
        sec_der = np.gradient(der) 
        sec_der = (sec_der**2) * np.sign(sec_der)
        sec_der = moving_average(sec_der,4)

        maxima,__ = find_peaks(sec_der,prominence=100000000000)
        minima,__ = find_peaks(sec_der*-1,prominence=100000000000)
        # prominences_maxima = peak_prominences(sec_der, maxima)[0]
        # print(prominences_maxima)
        # prominences_minima = peak_prominences(sec_der*-1, minima)[0]
        # print(prominences_minima)
        start = time.time()
        peaks,__ = find_peaks(binary_change,prominence = 1500000,plateau_size = 1)
        widths = peak_widths(binary_change,peaks,rel_height = 1)

        end = time.time() - start
        print(f'the big one took {end} seconds')

        mask = ~np.isin(peaks, test.trim_index+test.engage_index+test.retract_index)
        deposit_peaks = peaks[mask]

        print(test.engage_index)

        test_list = np.asarray(list(set(test.stub_indecies)))
        mask2 = ~np.isin(test_list, test.trim_index+test.engage_index+test.retract_index)
        test_list = test_list[mask2]
        test_list = test_list[(np.max(test.engage_index)<test_list)]



        print(deposit_peaks)
        print(test_list)

        print(f'during test detected {len(test_list)} stubs')
        print(f'after test detected {len(deposit_peaks)} stubs')
        

        # #print(peaks)
        # for peak in maxima:
        #     plt.plot(peak+1,binary_change[peak+1],'*',color='red')

        # for peak in minima:
        #     plt.plot(peak-1,binary_change[peak-1],'*',color = 'green')
        
        plt.subplot(1, 2, 1) 
        plt.plot(gaussian_filter1d(binary_change,2),color = 'green')
        plt.plot(binary_change,color = 'blue')
        for peak in deposit_peaks:
            plt.plot(peak,binary_change[peak],'*',color = 'black')
        plt.subplot(1, 2, 2) 
        plt.plot(binary_change,color = 'blue')
        for peak in test.stub_indecies:
            plt.plot(peak,binary_change[peak],'*',color = 'red')
        plt.tight_layout()
        plt.show()
        
        global_binary_change.append(binary_change)
        #global_median.append(moving_average(median,5))
        global_total_pix.append(moving_average(total_pix,8))
        global_total_intensity.append(total_intensity)


    f = open('test_data.txt',"w")
    for point in global_binary_change:
        f.write(f'{point}')
    f.close()
    # for idx,file in enumerate(files):
    # #     #plt.plot(vid)
    #     plt.plot(global_median[idx],label = f'{file}')
    #     #plt.plot(global_white_count_buffer[idx],label = f'deposit video {idx}')
    # plt.legend(loc="upper left")
    # plt.show()

    for idx,file in enumerate(files):
    #     #plt.plot(vid)
        plt.plot(global_total_pix[idx],label = f'{file}')
        #plt.plot(global_white_count_buffer[idx],label = f'deposit video {idx}')
    plt.legend(loc="upper left")
    plt.show()
    titles = ['Dripping Deposition','Stable Deposition','Oscillating Deposition']
    titles = ['Stable-Oscillating Deposition 1', 'Dripping Deposition','Stable Deposition 1', 'Stable-Oscillating Deposition 2', 'Oscillating Deposition', 'Stable-Oscillating Deposition 3', 'Stable Deposition 2']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplot(111)
    plt.legend(loc='upper right')

    for idx,threshold in enumerate(files):
    #     #plt.plot(vid)
        plt.plot(global_binary_change[idx],label = f'{titles[threshold-1]}')
    plt.legend(loc="upper left")

    plt.xlabel('Frame')
    plt.ylabel('Frame to Frame Pixel Difference')
    plt.title('Frame to Frame Pixel Difference for Stable and Oscillating Deposits')

    Vline = draggable_lines(ax, "h", 10000,len(max(global_binary_change, key=len)))
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

