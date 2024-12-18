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
import matplotlib.colors as mcolors
import sys
import glob
import os


warnings.filterwarnings("ignore", category=DeprecationWarning, message="The truth value of an empty array is ambiguous.")


class LiveWiggleometer: 

	def __init__(self, threshold):
		self.start_time = time.time()
		self.total_intensity_buffer = collections.deque(maxlen = 3)
		self.new_total_intensity_buffer = collections.deque(maxlen = 3)
		self.binary_image_ring_buffer = collections.deque(maxlen = 2)
		self.frame_change_buffer = collections.deque(maxlen = 3)
		self.binary_pixel_count_buffer = collections.deque(maxlen = 3)
		self.balling_data_buffer = collections.deque(maxlen=5)
		self.frame_change_buffer.append(1)
		self.stability_state = "Not Depositing"
		self.height,self.width = 1920,1080
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
		self.active_balling_idx = []
		roi = 0
		if roi != 0:
			self.x, self.y, self.w, self.h = roi
		else:
			self.x, self.y, self.w, self.h = 0,0,self.width,self.height
		


		#thresholding parameters
		self.deposit_state_threshold = 3087626
		self.balling_threshold = 11672639
		self.stubbing_threshold = 801825
		self.blue_threshold = 4000000


	def get_frame(self,frame):
		self.frame = frame
		self.frame_idx+= 1
		return 
		
	def gray_threshold(self):
		
		self.gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) 
		self.gray_image = cv2.GaussianBlur(src=self.gray_image, ksize=(5, 5), sigmaX=0.5)

		self.gray_image = self.gray_image[self.y:self.y+self.h, self.x:self.x+self.w]

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
		init_offset = 8
		stub_timing_factor = 1.2
		balling_padding = 11
		
		self.active_balling = False
		#print(self.stub_count)
		
		
		if np.mean(self.new_total_intensity_buffer) > self.balling_threshold or self.new_total_intensity > self.balling_threshold:
			self.stability_state = 'Balling'
			self.stub_frequency_buffer[-1] = 0
			#self.frame_change_buffer[-1] = 0
			self.balling_offset = 0
			if np.mean(np.asarray(self.balling_data_buffer)) > self.blue_threshold:
				self.active_balling = True
				self.active_balling_idx.append(self.frame_idx)
			
			self.frame_change_buffer[-1] = 0


		else:
			peaks,__ = find_peaks(np.asarray(self.stub_frequency_buffer),prominence = 200000,plateau_size = 1,width=1,height=500000)
			self.balling_offset += 1
			if self.balling_offset >= balling_padding:

				if len(peaks) != 0:
					if (self.frame_idx-(10-peaks[0])) not in (self.stub_indecies):
						self.stub_indecies.append(self.frame_idx-(10-peaks[0]))
						self.local_stub_indecies.append(self.frame_idx-(10-peaks[0]))
						self.active_stubbing = True
						#print('EVENT event')
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


	def get_deposit_state(self):
		#find trim state

		if self.deposit_state == 'Initalize' and self.frame_change_difference > self.deposit_state_threshold:
			self.deposit_state = 'Trim'
			print('trim')
			
		
		elif self.deposit_state == 'Trim' and self.frame_change < 1000:
			self.deposit_state = 'Awaiting Deposition'
			print('waiting')
			

		elif self.deposit_state == 'Awaiting Deposition' and self.frame_change_difference > self.deposit_state_threshold:
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

		if contours:
			largest_contour = max(contours, key=cv2.contourArea)
			hull = cv2.convexHull(largest_contour)

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
		if new_der[i-1] * new_der[i] < 0:  # Sign change event
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
					
			lengths.append((x2_pos-x1_pos))
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

	print(f"total number of depositon frames was {len(deposit_data['stability_states'])}")

	stable_percent = number_stable*100/len(deposit_data['stability_states'])
	balling_percent = number_balls*100/len(deposit_data['stability_states'])
	stubbing_percent = number_stubs*100/len(deposit_data['stability_states'])

	print(f"State Stable: {stable_percent:.2f}% of the video")
	print(f"State Balling: {balling_percent:.2f}% of the video")
	print(f"State Stubbing: {stubbing_percent:.2f}% of the video")
	print(f"Total Unstable: {stubbing_percent+balling_percent:.2f}% of video")


	print(f"During testing {(deposit_data['total_stub_occurances'])} stub events were detected\n")

	#print(deposit_data['stability_states'])
	return balling_percent,stubbing_percent

def create_heat_map(binary_change,deposit_data,drip_indecies):
	y_max = 1691315.0
	y_min = 44607.0

	# Normalize the time series between 0 and 1 for color mapping
	norm = mcolors.Normalize(vmin=y_min, vmax=y_max)

	# Create a colormap
	cmap = plt.get_cmap('OrRd')  # You can choose any colormap (e.g., 'viridis', 'plasma')

	# Create the figure and axis
	fig, ax = plt.subplots()

	# Loop through the time series and plot segments with continuous color changes
	for i in range(len(binary_change[deposit_data['deposit_start_idx']:deposit_data['deposit_end_idx']]) - 1):
		# Normalize the value to get a color from the colormap
		color = cmap(norm(binary_change[i+deposit_data['deposit_start_idx']]))
		
		# Plot the bead-like segment
		ax.plot([i, i+1], [0, 0], color=color, lw=10, solid_capstyle='round')

	# Hide axes for visual effect
	for drip in drip_indecies:
		plt.plot(drip,0,'o',markersize=20,color='orange')

	ax.set_axis_off()
	ball_size_factor = 11
	x_position = 19
	plt.show()

def get_drip_indecies(drip_events,deposit_data):
	if len(drip_events)>0:
		drip_events = np.asarray(drip_events)

		print(drip_events[np.where(np.diff(drip_events)>1)[0]])

		event_indecies = drip_events[np.where(np.diff(drip_events)>1)[0]]
		event_indecies = np.append(event_indecies,drip_events[-1])

		if 'Balling' in deposit_data['stability_states'][-5:-1]:
			event_indecies = event_indecies[:-1]
		event_indecies = event_indecies - np.max(deposit_data['deposit_start_idx'])
	else:
		event_indecies = []
	
	return event_indecies




def get_roi(file):
	cap = cv2.VideoCapture(file)

	if not cap.isOpened():
		print("Error: Could not open video.")
		exit()
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	if total_frames <= 0:
		print("Error: Video contains no frames or could not retrieve frame count.")
		cap.release()
		exit()
	cv2.namedWindow("Video Frame")
	cv2.createTrackbar("Frame", "Video Frame", 0, total_frames - 1, lambda x: None)
	while True:
		current_frame = cv2.getTrackbarPos("Frame", "Video Frame")
		cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
		ret, frame = cap.read()
		
		if ret:
			cv2.putText(frame, "Press 'r' to select ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
			cv2.imshow("Video Frame", frame)

		key = cv2.waitKey(30) & 0xFF
		
		if key == ord('r'):  
			roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
			cv2.destroyWindow("Select ROI")  
			print(f"Selected ROI: {roi}")
			break  

		elif key == ord('q'):  
			break

	cap.release()
	cv2.destroyAllWindows()
	return roi

if __name__ == '__main__':
	font = cv2.FONT_HERSHEY_SIMPLEX
	fourcc = cv2.VideoWriter_fourcc(*'H264')

	videos = []


	engage_pad = 7
	retract_pad = 10
	threshold = 125
	
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
	final_info = []
	roi_request = False

	
	

	#if you pass in a file, it will run one video. if you pass in a folder, it will run all videos.  
	#if you pass in --roi=True it will allow you to select an roi for the video
	for arg in sys.argv[1:]:

		if os.path.isdir(arg):
			path = sys.argv[1]
			for filename in glob.glob(os.path.join(path, '*.mp4')):
				bead_number = filename.split('_')[1].split('.')[0]
				with open(filename, 'r') as f:
					name = filename.split('/')[-1].split('.mp4')[0]
					videos.append([filename,name])
					

		elif os.path.isfile(arg):
			with open(sys.argv[1], 'r') as f:
				name = sys.argv[1].split('/')[-1].split('.mp4')[0]
				videos.append([sys.argv[1],name])

		if arg.startswith('--roi='):
			roi_value = arg.split("=")[1].lower()  # Extract the value after "="
			if roi_value == "true":
				roi_request = True
			elif roi_value == "false":
				roi_request = False			


	if roi_request:
		roi = get_roi(videos[0][0])
	else:
		roi = 0

	for i,file in enumerate(videos):
		print(f'the file is {file[0]}')
		test = LiveWiggleometer(file[0],threshold,roi)
		
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
		
		deposit_data = {'name': '',
						'stability_states': [],
						'stub_indecies':[],
						'stub_lengths': [],
						'stub_intensities':[],
						'deposit_length':0,
						'total_stub_events':0,
						'deposit_start_idx':0,
						'deposit_end_idx':0
						
		}

		stub_idx = 0
		plt.ion()
		cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
		cv2.resizeWindow("Test Window", 960, 540)
		i = 0


		while test.state:
			i+= 1
			
			
			start= time.time()

			test.classification_analysis()
				   
			test.get_deposit_state()

			test.get_stability_state()
			test.save_balling_data()


			#display_img = cv2.putText(test.frame,test.stability_state,(10,140), font, 2, (255,255,255), 2, cv2.LINE_AA)
			#display_img = cv2.putText(display_img,test.deposit_state,(10,200), font, 2, (255,255,255), 2, cv2.LINE_AA)

			if test.active_stubbing:
				pass#display_img = cv2.putText(display_img,'OSCILLATION DETECTED',(1200,200), font, 2, (255,255,255), 2, cv2.LINE_AA)

			if test.active_balling:
				pass#display_img = cv2.putText(display_img,'CURRENTLY BALLING',(1200,200), font, 2, (255,255,255), 2, cv2.LINE_AA)

			
			binary_change.append(np.mean(np.asarray(test.frame_change_buffer,dtype=np.float64)))
			true_binary_change.append(np.sum(test.binary_image-test.binary_image_ring_buffer[0]))

			total_pix.append(np.sum(test.binary_image))

			#cv2.rectangle(test.gray_image, (int(roi[0]), int(roi[1])), (int(roi[0]+roi[2]), int(roi[1]+roi[3])), (255, 255, 255), 2) 
			#if roi != 0:
				#cv2.imshow('binary frame',test.binary_image)
			cv2.imshow('Test Window',test.frame)
			cv2.waitKey(5)
			test.get_frame()
			plt.clf()
			if i <= 200:
				plt.plot(balling_data)
			if i > 200:
				plt.plot(balling_data[190:])
			plt.xlabel('Frame')
			plt.ylabel('Pixel Count (Pixels)')
			plt.ylim(-5000000,58000000)
			plt.draw()
			plt.pause(0.005)
			
			total_average_intensity.append(np.mean(np.asarray(test.total_intensity_buffer,dtype=np.float64)))
			total_intensity.append(test.total_intensity)
			balling_data.append(test.balling_data_plot)
			new_total_intensity.append(np.mean(test.new_total_intensity_buffer))
			if test.deposit_state == 'Depositing':
				stability_states.append(test.stability_state)



		#plt.show()

			

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
		

		# plt.plot(np.linspace(0,22,len(binary_change)),binary_change)
		# plt.xlabel('Video Frame (Seconds)')
		# plt.ylabel('Binary Change Intensity (Pixels)')
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
		deposit_data.update({'deposit_start_idx':np.max(test.engage_index)+engage_pad})
		deposit_data.update({'deposit_end_idx':np.min(test.retract_index)-retract_pad})
		deposit_data.update({'name':file[1]})

		drip_indecies = get_drip_indecies(test.active_balling_idx,deposit_data)
		create_heat_map(binary_change,deposit_data,drip_indecies)
		print(test.retract_index)
		print(test.active_balling_idx)
		print(stability_states)

		#print_stub_summary(deposit_data)
		#print(stability_states)

		balling_percent,stubbing_percent = print_general_deposit_information(deposit_data)
		
		#print_stub_summary(deposit_data)

  
		
		live_deposit_peaks = np.asarray(test.stub_indecies)
		mask2 = ~np.isin(live_deposit_peaks, test.trim_index+test.engage_index+test.retract_index)
		live_deposit_peaks = live_deposit_peaks[mask2]
		live_deposit_peaks = live_deposit_peaks[(np.max(test.engage_index) < live_deposit_peaks) | (live_deposit_peaks < np.min(test.retract_index))]
		plt.plot(binary_change)
		for peak in live_deposit_peaks:
			plt.plot(peak,binary_change[peak],'*k')

		

		string_to_binary = {
			'Stubbing': 1,  # Map 'StateA' to 1
			'Stable': 0 ,  # Map 'StateB' to 0
			'Balling': 2
		}
	
		global_binary_change.append(binary_change)
		global_total_pix.append(moving_average(total_pix,8))
		global_total_intensity.append(total_intensity)
		global_total_average_intensity.append(total_average_intensity)
		global_true_binary_change.append(true_binary_change)
		global_balling_data.append(balling_data)
		global_new_total_intensity.append(new_total_intensity)
		global_unstable_time.append([name,balling_percent,stubbing_percent])

		print([deposit_data['name'],balling_percent,stubbing_percent])

		plt.plot(binary_change[deposit_data['deposit_start_idx']:deposit_data['deposit_end_idx']])
		#plt.show()

		# y_min = np.min(binary_change[deposit_data['deposit_start_idx']:deposit_data['deposit_end_idx']])  # Minimum value for the y-axis
		# y_max = np.max(binary_change[deposit_data['deposit_start_idx']:deposit_data['deposit_end_idx']])  # Maximum value for the y-axis


		if balling_percent > 0:
			final_state = 'fail balling'
		elif stubbing_percent > 0:
			final_state = 'fail stubbing'
		else:
			final_state = 'pass'

		final_info.append([deposit_data['name'],final_state,balling_percent,stubbing_percent])

	#print(global_unstable_time)
	final_info = sorted(final_info, key=lambda x: x[0])

	with open("output.txt", "w") as file:
		for info in final_info:
			file.write(str(info)+'\n')
	

	# for idx,threshold in enumerate(files):
	# #	 #plt.plot(vid)
	#	 print(f'{titles[threshold-1]}')
	#	 plt.plot(global_new_total_intensity[idx],label = f'{titles[threshold-1]}')
	# plt.legend(loc="upper left")

	# plt.xlabel('Frame')
	# plt.ylabel('Total Pixel Intensity')
	# plt.title('Total Pixel Intensity for Various Deposition States')

	# Vline = draggable_lines(ax, "h", 10000,len(max(global_binary_change, key=len)))
	# # Update the legend after adding the draggable line
	# handles, labels = ax.get_legend_handles_labels()
	# ax.legend(handles, labels, loc='upper right')
	# plt.show()

	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# plt.subplot(111)
	# plt.legend(loc='upper right')
	# plt.legend(fontsize='10')

	# SMALL_SIZE = 8
	# MEDIUM_SIZE = 10
	# BIGGER_SIZE = 16



	# plt.rcParams.update({'font.size': '15'})
	# plt.rc('font', size=BIGGER_SIZE)		  # controls default text sizes
	# plt.rc('axes', titlesize=BIGGER_SIZE)	 # fontsize of the axes title
	# plt.rc('axes', labelsize=BIGGER_SIZE)	# fontsize of the x and y labels
	# plt.rc('xtick', labelsize=BIGGER_SIZE)	# fontsize of the tick labels
	# plt.rc('ytick', labelsize=BIGGER_SIZE)	# fontsize of the tick labels
	# plt.rc('legend', fontsize=MEDIUM_SIZE)	# legend fontsize

	# #plt.show()