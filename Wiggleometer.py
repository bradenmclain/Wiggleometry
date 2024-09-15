import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import collections
from scipy.stats import gmean
from draggable_lines import draggable_lines
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from scipy.optimize import root_scalar
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning, message="The truth value of an empty array is ambiguous.")


class Wiggleometer: 

    def __init__(self, path,threshold):
        self.start_time = time.time()
        self.video = cv2.VideoCapture(path)
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.state, self.frame = self.video.read()
        self.total_intensity_buffer = collections.deque(maxlen = 3)
        self.new_total_intensity_buffer = collections.deque(maxlen = 3)
        self.binary_image_ring_buffer = collections.deque(maxlen = 2)
        self.frame_change_buffer = collections.deque(maxlen = 3)
        self.binary_pixel_count_buffer = collections.deque(maxlen = 3)
        self.balling_data_buffer = collections.deque(maxlen=5)
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
        self.stability_buffer = collections.deque(maxlen = 4)
        self.stub_indecies = []
        self.frame_idx = 0
        self.pixel_count = 0
        self.trim_index = []
        self.engage_index = []
        self.retract_index = []
        self.new_total_intensity = []
        self.active_stubbing = False
        self.active_balling = False
        self.local_stub_indecies = []
        self.balling_data = 0
        self.balling_offset = 5
        


        #thresholding parameters
        self.deposit_state_threshold = 3587626
        self.balling_threshold = 11672639
        self.stubbing_threshold = 801825
        self.blue_threshold = 3220484
        11672639


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

        self.red = self.frame[:, :, 2]
        self.green = self.frame[:,:,1]
        self.blue = self.frame[:,:,0]
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

        _, self.binary_image = cv2.threshold(self.gray_image, self.threshold, 255, cv2.THRESH_BINARY)
        mask = self.gray_image >= self.threshold

        # Apply the mask: values above the threshold remain unchanged, others are set to 0
        self.threhold_image_without_background = np.where(mask, self.gray_image, 0)
        self.new_total_intensity = np.sum(self.threhold_image_without_background)


        binary_pixel_count = np.sum(self.binary_image)
        self.binary_pixel_count_buffer.append(binary_pixel_count)

        # self.white = np.zeros_like(self.gray_image)
        # self.white[np.where(self.gray_image>250)] = 255

    
    def threshold_image_by_rgb(self,frame, lower_rgb, upper_rgb):
        # Apply the threshold using the given RGB ranges
        lower_bound = np.array(lower_rgb, dtype="uint8")
        upper_bound = np.array(upper_rgb, dtype="uint8")
        mask = cv2.inRange(frame, lower_bound, upper_bound)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        return result, mask
    
    
    def save_balling_data(self):
        r_min = 255
        r_max = 255
        g_min = 255
        g_max = 255
        b_min = self.threshold
        b_max = 254

        # Set the lower and upper RGB boundaries based on trackbar positions
        lower_rgb = [b_min, g_min, r_min]
        upper_rgb = [b_max, g_max, r_max]

        # Apply the threshold to the current frame
        thresholded_frame, mask = self.threshold_image_by_rgb(self.frame, lower_rgb, upper_rgb)
        self.balling_data = np.sum(thresholded_frame)

        self.balling_data_buffer.append(self.balling_data)
        self.balling_data_plot = np.mean(np.asarray(self.balling_data_buffer))
        


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
        self.new_total_intensity_buffer.append(self.new_total_intensity)
        self.frame_change = np.sum(self.binary_image - self.binary_image_ring_buffer[0],dtype=np.float64)
        self.frame_change_buffer.append(self.frame_change)
        self.frame_change_difference = np.mean(np.asarray(self.frame_change_buffer,dtype=np.float64))
        self.stub_frequency_buffer.append(self.frame_change_difference)

    def get_stability_state(self):
        self.active_stubbing = False
        init_offset = 6
        stub_timing_factor = 1.1
        balling_padding = 7
        
        self.active_balling = False
        #print(self.stub_count)
        

        if self.deposit_state != 'Depositing':
            self.stability_state = 'Not Depositing'

        
        elif np.mean(self.new_total_intensity_buffer) > self.balling_threshold or self.new_total_intensity > self.balling_threshold:
            self.stability_state = 'Balling'
            self.stub_frequency_buffer[-1] = 0
            #self.frame_change_buffer[-1] = 0
            self.balling_offset = 0
            if np.mean(np.asarray(self.balling_data_buffer)) > self.blue_threshold:
                self.active_balling = True
            
            #self.frame_change_buffer[-1] = 0


        else:
            peaks,__ = find_peaks(np.asarray(self.stub_frequency_buffer),prominence = 400000,plateau_size = 1,width=1)
            self.balling_offset += 1
            if self.balling_offset >= balling_padding:

                if len(peaks) != 0:
                    if (self.frame_idx-(10-peaks[0])) not in (self.stub_indecies):
                        self.stub_indecies.append(self.frame_idx-(10-peaks[0]))
                        self.local_stub_indecies.append(self.frame_idx-(10-peaks[0]))
                        self.active_stubbing = True
                        #print('EVENT DETECTED')
                    self.stability_state = 'Stubbing'


                if np.mean(np.asarray(self.frame_change_buffer,dtype=np.float64)) > self.stubbing_threshold:
                    self.stability_state = 'Stubbing'
                    self.stub_count +=1 
                    #print('stubbing from thresholding')
                elif len(self.local_stub_indecies) == 1:
                    if self.frame_idx <= self.local_stub_indecies[0] + init_offset:
                        self.stability_state = 'Stubbing' 
                        #print('stubbing from init')
                    else:
                        self.local_stub_indecies = []
                        #print('CLEARING LOCAL INDEX')
                elif len(self.local_stub_indecies) > 1:
                    if self.frame_idx<=(((self.local_stub_indecies[-1] - self.local_stub_indecies[-2]) * stub_timing_factor) + self.local_stub_indecies[-1]):
                        self.stability_state = 'Stubbing'
                        #print('stubbing from previous')
                        #print('Stubbing from previous stubs')
                    else:
                        self.local_stub_indecies = []
                        #print('CLEARING LOCAL INDEX')
                else:
                    self.stability_state = 'Stable'
                    self.local_stub_indecies = []
            
            else:
                self.stability_state = 'Stable'





    def adjust_threshold(self):
        pass
    
    
    def get_deposit_state(self):
        #find trim state

        if self.deposit_state == 'Initalize' and self.frame_change_difference > self.deposit_state_threshold:
            self.deposit_state = 'Trim'
            print('trim')
            
        
        if self.deposit_state == 'Trim' and self.frame_change < 1000:
            self.deposit_state = 'Awaiting Deposition'
            print('waiting')
            

        if self.deposit_state == 'Awaiting Deposition' and self.frame_change_difference > self.deposit_state_threshold:
            self.deposit_state = 'Engage'
            print('engage')

        if self.deposit_state == 'Engage' and self.frame_change_difference < self.deposit_state_threshold:
            self.deposit_state = 'Depositing'
            print('Deposit has started')

        if self.deposit_state == 'Depositing' and self.frame_change_difference > self.deposit_state_threshold and self.stability_state != 'Balling':
            self.deposit_state = 'Retract'
            print('OG retract')

        if self.deposit_state == 'Depositing' and self.frame_change_difference < 1000:
            self.deposit_state = 'Retract'
            print('NEW retract')

        elif self.deposit_state == 'Retract' and self.frame_change_difference < 1000:
            self.deposit_state = 'Deposition Complete'
            print('complete')


        #record when things happen
        if self.deposit_state == 'Trim':
            self.trim_index.append(self.frame_idx)
            
        
        if self.deposit_state == 'Engage':
            self.engage_index.append(self.frame_idx)
            self.stub_frequency_buffer[-1] = 0

        if self.deposit_state == 'Retract':
            self.retract_index.append(self.frame_idx)
            self.stub_frequency_buffer[-1] = 0


    def get_convex_hull(self):
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #Find the convex hull of the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)

            # Draw the convex hull on a new image
            hull_image = np.zeros_like(self.binary_image)
            cv2.drawContours(hull_image, [hull], 0, 255, thickness=cv2.FILLED)
            wire = hull_image - test.binary_image
    
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

def find_stub_indecies(binary_change,engage_index,retract_index):
    
    peaks,__ = find_peaks(binary_change,prominence = 400000,plateau_size = 1,width=1,height = test.stubbing_threshold)

    stubs = np.asarray(peaks)
    der = np.gradient((binary_change))

    x_org = np.arange(0,len(der))

    interp_function = interpolate.interp1d(x_org,binary_change)

    der_interp_function = interpolate.interp1d(x_org,der)
    xnew = np.arange(0,len(der)-1,.001)
    
    new_der = der_interp_function(xnew)
    new_sec_der = np.gradient(new_der)
    sec_der_interp_function = interpolate.interp1d(xnew,new_sec_der)

    zero_crossings = []
    der_peaks = []
    der_valleys = []

    for i in range(1, len(xnew)):
        if new_der[i-1] * new_der[i] < 0:  # Sign change detected
            # Step 2: Refine the zero crossing using root_scalar
            root_result = root_scalar(der_interp_function, bracket=[xnew[i-1], xnew[i]])
            if root_result.converged:
                zero_crossings.append((root_result.root))


 
    for zero_crossing in zero_crossings:
        if sec_der_interp_function(zero_crossing) >= 0:
            #plt.scatter(zero_crossing, der_interp_function(np.array(zero_crossing)), color='red', zorder=5, label='Zero Crossings')
            der_valleys.append(zero_crossing)
        else:
            #plt.scatter(zero_crossing, der_interp_function(np.array(zero_crossing)), color='blue', zorder=5, label='Zero Crossings')
            der_peaks.append(zero_crossing)


    if len(engage_index) > 0:
        stubs = stubs[(np.max(engage_index) < stubs)]
        

    if len(retract_index) > 0:
        stubs = stubs[(stubs < np.min(retract_index))]

    lengths = []
    positions = []

    for peak in stubs:

        left = (np.where(((np.diff(np.sign(der_valleys-peak)) != 0)*1)==1)[0])
        if left.size > 0:
            left = int(left[0])
        if left != None:
            right = (left + 1)

            local_x = np.arange((der_valleys[int(left)]),(der_valleys[int(right)]),.001)
            local_y = interp_function(local_x)

            #if the left point is higher up, use that and the right point locally horizontal to it
            if interp_function(der_valleys[int(left)]) > interp_function(der_valleys[int(right)]):
                new_point = (np.where(((np.diff(np.sign(interp_function(der_valleys[int(left)])-local_y)) != 0)*1)==1))
                if new_point[0].size > 1:
                    x1_pos = local_x[new_point[0][0]]
                    x2_pos = local_x[new_point[0][-1]]

                else:
                    x1_pos = der_valleys[int(left)]
                    x2_pos = local_x[new_point[0][0]]
            #if the right point is higher up, use that and the left point locally horizontal to it
            else:
                new_point = (np.where(((np.diff(np.sign(local_y-interp_function(der_valleys[int(right)]))) != 0)*1)==1))
                
                #if the newly found line crosses over itself, take the inside points
                if new_point[0].size > 1:
                    x1_pos = local_x[new_point[0][0]]
                    x2_pos = local_x[new_point[0][-1]]

                else:
                    x1_pos = local_x[new_point[0][0]]
                    x2_pos = der_valleys[int(right)]
                    
            lengths.append((x2_pos-x1_pos)/30)
            positions.append([x1_pos,x2_pos,interp_function(x1_pos),interp_function(x2_pos)]) 


    
    return stubs, lengths, positions

def print_stub_summary(deposit_data):
    print(f"During testing {(deposit_data['total_stub_occurances'])} stub events were detected")
    if (deposit_data['total_stub_occurances']) != 0:
        for i,stub in enumerate(deposit_data['stub_lengths']):
            print(f"Stub event recorded at {(deposit_data['stub_indecies'][i])/30:0.3f} seconds lasted for {stub:0.3f} seconds")

def print_general_deposit_information(deposit_data):
    print(f"\nFor deposit {deposit_data['name']}:")
    print(f"Total deposition time was {(deposit_data['deposit_length'])/30} seconds")
    
    number_balls = deposit_data['stability_states'].count('Balling')
    number_stubs = deposit_data['stability_states'].count('Stubbing')
    number_stable = deposit_data['stability_states'].count('Stable')

    stable_percent = number_stable*100/len(deposit_data['stability_states'])
    balling_percent = number_balls*100/len(deposit_data['stability_states'])
    stubbing_percent = number_stubs*100/len(deposit_data['stability_states'])

    print(f"State Stable: {stable_percent:.2f}% of the video")
    print(f"State Balling: {balling_percent:.2f}% of the video")
    print(f"State Stubbing: {stubbing_percent:.2f}% of the video")
    print(f"Total Unstable: {stubbing_percent+balling_percent:.2f}% of video")


    print(f"During testing {(deposit_data['total_stub_occurances'])} stub events were detected\n")

    #print(deposit_data['stability_states'])
    return stubbing_percent+balling_percent






if __name__ == '__main__':

    #test = Wiggleometer("./data/Second/wiggleometer_deposit_2.mp4")
    font = cv2.FONT_HERSHEY_SIMPLEX
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    #fourcc = -1

    engage_pad = 12
    retract_pad = 10
    
    i = 0
    global_binary_change = []
    global_median = []
    global_total_pix = []
    global_total_intensity = []
    global_deposition_data = []
    global_true_binary_change = []
    global_balling_data = []
    global_new_total_intensity = []
    global_unstable_time = []


    videos = {1500: [0,6,7],
              1450: [8,9,10],
              1400: [1,30,12],
              1350: [13,14,15],
              1300: [16,17,18,2],
              1250: [19,20,21],
              1200: [22,23,24],
              1150: [25,26,27],
              1100: [28,29,5],
              1000: [14,26]
    }

    #files = [file_1,file_2]
    #files = [1,2,3,4,5,6,7]
    roi = [795,444,305,588]
    threshold = 100
    #videos = np.arange(0,31)
    videos = [1]

    #for i,file_num  in enumerate(videos[1100]):
    for i,file_num in enumerate(videos):

        
        #file = f"./data/Stability_Experiment/trial_{file_num}.mp4"
        file = f"./data/Wiggleometer_Deposits/wiggleometer_deposit_{file_num}.mp4"
        print(f'the file is {file}')
        test = Wiggleometer(file,threshold)
        
        test.get_frame()
        height,width,val = test.frame.shape
        binary_change=[]
        blue_count = []
        green_count = []
        true_binary_change = []
        gray_change = []
        median = []
        total_average_intensity = []
        total_intensity = []
        new_total_intensity = []
        global_total_average_intensity = []
        balling_data = []
        stability_states = []
        total_pix = []
        frame = 0
        peak_indexs = []
        
        deposit_data = {'name': file.split("/")[-1].split(".")[0],
                        'stability_states': [],
                        'stub_indecies':[],
                        'stub_lengths': [],
                        'stub_intensities':[],
                        'deposit_length':0,
                        'total_stub_events':0
                        
        }

        stub_idx = 0
        
        # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("frame", (round(width/2), round(height/2)) )
        while test.state:
            
            start= time.time()

            test.classification_analysis()
                   
            test.get_deposit_state()

            test.get_stability_state()
            test.save_balling_data()
            #print(f'it took {time.time()-start} seconds')

            display_img = cv2.putText(test.frame,test.stability_state,(10,140), font, 2, (255,255,255), 2, cv2.LINE_AA)
            display_img = cv2.putText(display_img,test.deposit_state,(10,200), font, 2, (255,255,255), 2, cv2.LINE_AA)

            if test.active_stubbing:
                display_img = cv2.putText(display_img,'OSCILLATION DETECTED',(1200,200), font, 2, (255,255,255), 2, cv2.LINE_AA)

            if test.active_balling:
                display_img = cv2.putText(display_img,'CURRENTLY BALLING',(1200,200), font, 2, (255,255,255), 2, cv2.LINE_AA)


           
            #if test.deposit_state == 'Depositing':
            #     start = time.time()
            #     median.append(test.get_median_pixel_intensity())
                
            #     #total_intensity.append(np.sum(test.gray_image))
            #     print(f'that took {time.time()-start} seconds')
            
            binary_change.append(np.mean(np.asarray(test.frame_change_buffer,dtype=np.float64)))
            true_binary_change.append(np.sum(test.binary_image-test.binary_image_ring_buffer[0]))

            total_pix.append(np.sum(test.binary_image))
            #white_count.append(np.sum(test.white))


            cv2.rectangle(test.gray_image, (int(roi[0]), int(roi[1])), (int(roi[0]+roi[2]), int(roi[1]+roi[3])), (255, 255, 255), 2) 
            # cv2.imshow('frame',display_img)
            # cv2.waitKey(10)
            # print(test.stability_state)
            # print(test.stub_frequency_buffer)
            # print('its balling right the frick now')
            
            # print(f'intensity {test.total_intensity}')
            test.get_frame()
            
            total_average_intensity.append(np.mean(np.asarray(test.total_intensity_buffer,dtype=np.float64)))
            total_intensity.append(test.total_intensity)
            balling_data.append(test.balling_data_plot)
            new_total_intensity.append(np.mean(test.new_total_intensity_buffer))
            if test.deposit_state == 'Depositing':
                stability_states.append(test.stability_state)

            

        #print(peak_indexs)
        #plt.plot(moving_average(test.balling_data,5))
        #global_balling_data.append(test.balling_data)
        #plt.show()

        binary_change = np.asarray(binary_change)
        true_binary_change = np.asarray(true_binary_change)
        # plt.title('binary change')
        #plt.title('Frame to Frame Pixel Difference for Oscillating Deposition')
        # plt.xlabel('Frame')
        # plt.ylabel('Total Pixel Count')
        # plt.plot(binary_change,label = 'Stable')
        # plt.legend()
        # # # plt.plot(true_binary_change)
        # plt.show()

        
        
        #plt.plot(total_intensity,color='blue')
        #plt.plot(true_binary_change,color='red')
        #plt.show()
        stubs,lengths,positions = find_stub_indecies(binary_change,test.engage_index,test.retract_index)





        # #find_stub_lengths(binary_change,stubs,test.engage_index,test.retract_index)

        
        plt.plot(binary_change,color='blue')
        for position in positions:
            plt.plot([position[0],position[1]],[position[2],position[3]],color='black')


        #plt.plot(white_count)
        # plt.plot(moving_average(white_count,10),color = 'black')
        # plt.plot(moving_average(red_count,10),color = 'red')
        # plt.plot(moving_average(blue_count,10),color = 'blue')
        # plt.plot(moving_average(green_count,10),color = 'green')
        # plt.title('White Count')
        # plt.show()
        if engage_pad != 0:
            stability_states = stability_states[engage_pad:]

        if retract_pad != 0:
            stability_states = stability_states[:-retract_pad]
        
        
        deposit_data.update({'stability_states':stability_states})
        deposit_data.update({'stub_indecies':stubs})
        deposit_data.update({'stub_lengths':lengths})
        deposit_data.update({'stub_intensities':binary_change[stubs]})
        deposit_data.update({'deposit_length':len(stability_states)})
        deposit_data.update({'total_stub_occurances':len(stubs)})
        #print_stub_summary(deposit_data)
        print(stability_states)
        unstable_time = print_general_deposit_information(deposit_data)
        #print_stub_summary(deposit_data)

  
        
        live_deposit_peaks = np.asarray(test.stub_indecies)
        mask2 = ~np.isin(live_deposit_peaks, test.trim_index+test.engage_index+test.retract_index)
        live_deposit_peaks = live_deposit_peaks[mask2]
        live_deposit_peaks = live_deposit_peaks[(np.max(test.engage_index) < live_deposit_peaks) | (live_deposit_peaks < np.min(test.retract_index))]      

        for peak in live_deposit_peaks:
            plt.plot(peak,binary_change[peak],marker='*',color='black')
        #plt.plot(binary_change,color='blue')     
        plt.show() 

        # print(stubs)
        

        # #print(peaks)

        

        # plt.plot(total_average_intensity,color='red')
        # plt.plot(total_intensity,color='blue')
        # plt.show()
        global_binary_change.append(binary_change)
        global_total_pix.append(moving_average(total_pix,8))
        global_total_intensity.append(total_intensity)
        global_total_average_intensity.append(total_average_intensity)
        global_true_binary_change.append(true_binary_change)
        global_balling_data.append(balling_data)
        global_new_total_intensity.append(new_total_intensity)
        global_unstable_time.append([file_num,unstable_time])
        print([file_num,unstable_time])
    print(global_unstable_time)

    

    # for new_intensity in global_binary_change:
    #     plt.plot(new_intensity)
    # plt.show()

    # for new_intensity in global_total_intensity:
    #     plt.plot(new_intensity)
    # plt.show()


    # f = open('test_data.txt',"w")
    # for point in global_binary_change:
    #     f.write(f'{point}')
    # f.close()
    # for idx,file in enumerate(files):
    # #     #plt.plot(vid)
    #     plt.plot(global_median[idx],label = f'{file}')
    #     #plt.plot(global_white_count_buffer[idx],label = f'deposit video {idx}')
    # plt.legend(loc="upper left")
    # plt.show()

    # for idx,file in enumerate(files):
    # #     #plt.plot(vid)
    #     plt.plot(global_binary_change[idx],label = f'{file}')
    #     #plt.plot(global_white_count_buffer[idx],label = f'deposit video {idx}')
    # plt.legend(loc="upper left")
    # #plt.show()
    # titles = ['Dripping Deposition','Stable Deposition','Oscillating Deposition']
    # titles = ['Stable-Oscillating Deposition 1', 'Dripping Deposition','Stable Deposition 1', 'Stable-Oscillating Deposition 2', 'Oscillating Deposition', 'Stable-Oscillating Deposition 3', 'Stable Deposition 2']

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.subplot(111)
    # plt.legend(loc='upper right')

    

    # for idx,threshold in enumerate(files):
    # #     #plt.plot(vid)
    #     print(f'{titles[threshold-1]}')
    #     plt.plot(global_new_total_intensity[idx],label = f'{titles[threshold-1]}')
    # plt.legend(loc="upper left")

    # plt.xlabel('Frame')
    # plt.ylabel('Total Pixel Intensity')
    # plt.title('Total Pixel Intensity for Various Deposition States')

    # Vline = draggable_lines(ax, "h", 10000,len(max(global_binary_change, key=len)))
    # # Update the legend after adding the draggable line
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, loc='upper right')
    # plt.show()

    # plt.show()
    # for idx,threshold in enumerate(files):
    # #     #plt.plot(vid)
    #     plt.plot(global_balling_data[idx],label = f'{titles[threshold-1]}')
    # plt.legend(loc="upper left")

    # plt.xlabel('Frame')
    # plt.ylabel('Total Blue Pixel Count')
    # plt.title('Frame to Frame Blue Pixel Count for Various Deposition States')

    # Vline = draggable_lines(ax, "h", 10000,len(max(global_total_average_intensity, key=len)))
    # # Update the legend after adding the draggable line
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, loc='upper right')
    # plt.show()
        
        # plt.plot(pixel_count_array)
        # plt.show()
    # while Vline.XorY > 0:
    #     handles, labels = ax.get_legend_handles_labels()
    #     print(f'the position is {Vline.XorY}')
    #     plt.draw()
    #     plt.pause(.1)

    #     plt.show()

