import LiveWiggleometer
import cv2
import sys
import os
import glob
import numpy as np
import time

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
			
			total_average_intensity.append(np.mean(np.asarray(test.total_intensity_buffer,dtype=np.float64)))
			total_intensity.append(test.total_intensity)
			balling_data.append(test.balling_data_plot)
			new_total_intensity.append(np.mean(test.new_total_intensity_buffer))
			if test.deposit_state == 'Depositing':
				stability_states.append(test.stability_state)



	
