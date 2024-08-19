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
        self.active_event = False
        self.local_stub_indecies = []
        


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
        self.stub_frequency_buffer.append(np.mean(np.asarray(self.frame_change_buffer,dtype=np.float64)))

    def get_stability_state(self):
        self.active_event = False
        init_offset = 8
        stub_timing_factor = 1.3
        #print(self.stub_count)
        if self.deposit_state != 'Depositing':
            self.stability_state = 'Not Depositing'

        elif np.mean(self.total_intensity_buffer) > self.balling_threshold:
            self.stability_state = 'Balling'

        else:
            peaks,__ = find_peaks(np.asarray(self.stub_frequency_buffer),prominence = 500000,plateau_size = 1,width=1)

            if len(peaks) != 0:
                if (self.frame_idx-(10-peaks[0])) not in (self.stub_indecies):
                    self.stub_indecies.append(self.frame_idx-(10-peaks[0]))
                    self.local_stub_indecies.append(self.frame_idx-(10-peaks[0]))
                    self.active_event = True
                    print('EVENT DETECTED')
                self.stability_state = 'Stubbing'

            if np.mean(np.asarray(self.frame_change_buffer,dtype=np.float64)) > self.stubbing_threshold:
                self.stability_state = 'Stubbing'
                self.stub_count +=1 
                print('stubbing from thresholding')
            elif len(self.local_stub_indecies) == 1:
                if self.frame_idx <= self.local_stub_indecies[0] + init_offset:
                    self.stability_state = 'Stubbing' 
                    print('stubbing from init')
                else:
                    self.local_stub_indecies = []
                    print('CLEARING LOCAL INDEX')
            elif len(self.local_stub_indecies) > 1:
                if self.frame_idx<=(((self.local_stub_indecies[-1] - self.local_stub_indecies[-2]) * stub_timing_factor) + self.local_stub_indecies[-1]):
                    self.stability_state = 'Stubbing'
                    print('stubbing from previous')
                    #print('Stubbing from previous stubs')
                else:
                    self.local_stub_indecies = []
                    print('CLEARING LOCAL INDEX')
            else:
                self.stability_state = 'Stable'
                self.local_stub_indecies = []





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
    
    peaks,__ = find_peaks(binary_change,prominence = 1000000,plateau_size = 1,width=1,height = test.stubbing_threshold)
    prominences = peak_prominences(binary_change, peaks)

    stubs = np.asarray(peaks)
    der = np.gradient((binary_change))
    sec_der = np.gradient(der) 

    x_org = np.arange(0,len(der))

    interp_function = interpolate.interp1d(x_org,binary_change)
    inverse_interp_function = interpolate.interp1d(binary_change,x_org)

    der_interp_function = interpolate.interp1d(x_org,der)
    xnew = np.arange(0,len(der)-1,.01)
    
    new_der = der_interp_function(xnew)
    new_sec_der = np.gradient(new_der)
    sec_der_interp_function = interpolate.interp1d(xnew,new_sec_der)

    zero_crossings = []
    der_peaks = []
    der_valleys = []

    for i in range(1, len(x_org)):
        if der[i-1] * der[i] < 0:  # Sign change detected
            # Step 2: Refine the zero crossing using root_scalar
            root_result = root_scalar(der_interp_function, bracket=[x_org[i-1], x_org[i]])
            if root_result.converged:
                zero_crossings.append(root_result.root)

 
    #plt.plot(x_org,der,color = 'blue')
    # plt.plot(peaks, binary_change[peaks], "x")
    # plt.plot(binary_change,color = 'green')
    # plt.axhline(0, color='black', linestyle='--', lw=0.8)  # Add a horizontal line at y=0
    for zero_crossing in zero_crossings:
        if sec_der_interp_function(zero_crossing) >= 0:
            #plt.scatter(zero_crossing, der_interp_function(np.array(zero_crossing)), color='red', zorder=5, label='Zero Crossings')
            der_valleys.append(zero_crossing)
        else:
            #plt.scatter(zero_crossing, der_interp_function(np.array(zero_crossing)), color='blue', zorder=5, label='Zero Crossings')
            der_peaks.append(zero_crossing)



        

    # plt.legend()
    # plt.show()

    # mask = ~np.isin(post_process_deposit_peaks, trim_index+engage_index+retract_index)
    # stubs = peaks[mask]

    if len(engage_index) > 0:
        stubs = stubs[(np.max(engage_index) < stubs)]
        

    if len(retract_index) > 0:
        stubs = stubs[(stubs < np.min(retract_index))]


    for peak in stubs:

        left = int(np.where(((np.diff(np.sign(der_valleys-peak)) != 0)*1)==1)[0])
        if left != None:
            right = int(left + 1)
            #print(peak)
            #print(der_valleys[int(left)],der_valleys[int(right)])
            peak_time = der_valleys[int(right)] - der_valleys[int(left)]
            print(f'peak lasted {peak_time/30} seconds')
            # plt.plot(der_valleys[int(left)], interp_function(der_valleys[int(left)]), color='red', marker=".")
            # plt.plot(der_valleys[int(right)], interp_function(der_valleys[int(right)]), color='green', marker='.')
            # plt.plot(peak,interp_function(peak),marker='.',color='black')

            local_ynew = np.arange(interp_function(der_valleys[int(right)]),interp_function(der_valleys[int(left)]),.1)
            #print(local_ynew)
            #print(f'the length is {len(local_ynew)}')


            if interp_function(der_valleys[int(left)]) > interp_function(der_valleys[int(right)]):
                local_ynew = np.linspace(interp_function(round(der_valleys[int(right)])),interp_function(round(der_valleys[int(left)])),20)
                local_inverse_interp_function = interpolate.interp1d(binary_change[round(der_valleys[int(left)]):round(der_valleys[int(right)])],x_org[round(der_valleys[int(left)]):round(der_valleys[int(right)])])

                vals = local_inverse_interp_function(local_ynew)
                zero_vals = (vals- der_valleys[int(left)])
                plt.plot(zero_vals)
                plt.show()
                pass
            else:
                #print('right was greater')
                pass


    plt.plot(binary_change)
    plt.show()



    results_half = peak_widths(binary_change, peaks, rel_height=0.5)
    results_full = peak_widths(binary_change, peaks, rel_height=1, prominence_data=prominences)
    plt.hlines(*results_half[1:], color="C2")
    plt.plot(binary_change)
    plt.plot(xnew,new_der,color = 'green')
    plt.plot(peaks, binary_change[peaks], "x")
    plt.plot()
    
    
    return stubs

def find_stub_lengths(binary_change,stub_indecies,engage_index,retract_index):
    der = np.gradient((binary_change))
    sec_der = np.gradient(der) 
    #sec_der = (sec_der**2) * np.sign(sec_der)
    #sec_der = moving_average(sec_der,4)

    maxima,__ = find_peaks(sec_der,prominence=10000000000)
    minima,__ = find_peaks(sec_der*-1,prominence=10000000000)
    maxima_prominences = peak_prominences(sec_der, maxima)[0]

    maxima = np.asarray(maxima)
    minima = np.asarray(minima)
    print(f'the original length was {len(maxima)}')



    if len(engage_index) > 0:
        maxima = maxima[(np.max(engage_index) < maxima)]
        

    if len(retract_index) > 0:
        maxima = maxima[(maxima < np.min(retract_index))]
    print(f' the trimmed length is {len(maxima)}')

    if len(engage_index) > 0:
        minima = minima[(np.max(engage_index) < minima)]
        

    if len(retract_index) > 0:
        minima = minima[(minima < np.min(retract_index))]
    print(f' the trimmed length is {len(maxima)}')

    for peak in maxima:
        plt.plot(peak,binary_change[peak],'*',color='red')

    for peak in minima:
        plt.plot(peak,binary_change[peak],'*',color='green')
    
    plt.plot(binary_change)
    plt.plot(sec_der,color='orange')

    # for peak in minima:
    #     plt.plot(peak-1,binary_change[peak-1],'*',color = 'green')


    deposit_maxima = []



    for peak in stub_indecies:
        plt.plot(peak,binary_change[peak],'*',color = 'black')

    plt.show()
    
    
    pass

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
    global_deposition_data = []
    videos = [1,2,3,4,5,6,7]
    files = [4,5,6]
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
        deposit_data = {'stub_indecies':[],
                        'stub_lengths': [],
                        'stub_intensities':[],
                        'length':0,
                        'total_stub_events':0
                        
        }

        stub_idx = 0
        

        while test.state:
            start = time.time()
            test.classification_analysis()
            test.get_deposit_state()
            test.get_stability_state()
            display_img = cv2.putText(test.frame,test.stability_state,(10,140), font, 2, (255,255,255), 2, cv2.LINE_AA)
            display_img = cv2.putText(display_img,test.deposit_state,(10,200), font, 2, (255,255,255), 2, cv2.LINE_AA)

            if test.active_event:
                display_img = cv2.putText(display_img,'OSCILLATION DETECTED',(1200,200), font, 2, (255,255,255), 2, cv2.LINE_AA)


           
            #if test.deposit_state == 'Depositing':
            #     start = time.time()
            #     median.append(test.get_median_pixel_intensity())
                
            #     #total_intensity.append(np.sum(test.gray_image))
            #     print(f'that took {time.time()-start} seconds')
            
            binary_change.append(np.mean(np.asarray(test.frame_change_buffer,dtype=np.float64)))
            total_pix.append(np.sum(test.binary_image))
            cv2.rectangle(test.gray_image, (int(roi[0]), int(roi[1])), (int(roi[0]+roi[2]), int(roi[1]+roi[3])), (255, 255, 255), 2) 
            # cv2.imshow('frame',display_img)
            # cv2.waitKey(30)

            test.get_frame()
            
            total_intensity.append(np.mean(np.asarray(test.total_intensity_buffer,dtype=np.float64)))
            

        #print(peak_indexs)

        binary_change = np.asarray(binary_change)
        stubs = find_stub_indecies(binary_change,test.engage_index,test.retract_index)

        #find_stub_lengths(binary_change,stubs,test.engage_index,test.retract_index)

        
        plt.plot(binary_change,color='blue')
        plt.show()
        deposit_data.update({'stub_indecies':stubs})
        deposit_data.update({'stub_intensities':binary_change[stubs]})
        deposit_data.update({'length':((np.max(test.retract_index)-np.min(test.engage_index))/30)})
        deposit_data.update({'total_stub_occurances':len(stubs)})

   
        #test_list = np.asarray(list(set(test.stub_indecies)))
        live_deposit_peaks = np.asarray(test.stub_indecies)
        mask2 = ~np.isin(live_deposit_peaks, test.trim_index+test.engage_index+test.retract_index)
        live_deposit_peaks = live_deposit_peaks[mask2]
        live_deposit_peaks = live_deposit_peaks[(np.max(test.engage_index) < live_deposit_peaks) | (live_deposit_peaks < np.min(test.retract_index))]        

        print(f'during test detected {len(live_deposit_peaks)} stubs')
        print(f'after test detected {len(stubs)} stubs')

        print(f'the deposition lasted {((np.max(test.retract_index)-np.min(test.engage_index))/30)} seconds')

        print(live_deposit_peaks)
        print(stubs)
        

        # #print(peaks)

        

        
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
    # while Vline.XorY > 0:
    #     handles, labels = ax.get_legend_handles_labels()
    #     print(f'the position is {Vline.XorY}')
    #     plt.draw()
    #     plt.pause(.1)

    #     plt.show()

