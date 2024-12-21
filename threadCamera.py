import cv2
from multiprocessing import Process, Queue, Value, Manager,Lock
from ctypes import c_char_p
import time
import cherrypy
from icecream import ic
import numpy as np
from LiveWiggleometer import LiveWiggleometer
import os

class VideoDisplay:
	##### Video States
	### 0 is idle
	### 1 is recording
	### 2 is stop recording
	### 3 is close program
	def __init__(self, video_source=0):
		self.display_queue = Queue(maxsize=10)  # Queue to hold frames
		self.record_queue = Queue(maxsize=600)
		self.close_video = Value('b', False)  # Shared flag to stop processes
		self.state_flag = Value('i', 0)
		self.wiggleometer_flag = Value('i', 0)
		self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

		self.fps = 30
		self.height = 720
		self.width = 1280

		self.wiggleometer = LiveWiggleometer(120)


	def capture_frames(self):
		self.capture = cv2.VideoCapture(0)
		self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
		self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
		self.capture.set(cv2.CAP_PROP_FPS, 30)  
		
		self.state, self.frame = self.capture.read()
		self.height, self.width, rgb= self.frame.shape
		self.fps = self.capture.get(cv2.CAP_PROP_FPS)

		print(f'height is {self.height}')
		print(f'width is {self.width}')
		print(f'fps is {self.fps}')

		while True:
				start_time = time.time()
				ret, frame = self.capture.read()
				if ret:
					if self.state_flag.value == 1:
						self.record_queue.put(frame)
					sleep = (1/self.fps) - (time.time() - start_time)
					if self.state_flag.value == 2:
						self.close_video.value = True
					if self.wiggleometer_flag.value == 1:
						self.wiggleometer.get_frame(frame)
						self.wiggleometer.classification_analysis()
						self.wiggleometer.get_deposit_state()
						self.wiggleometer.get_stability_state()
						self.wiggleometer.save_balling_data()
						frame = cv2.putText(frame,self.wiggleometer.stability_state,(10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
					self.display_queue.put(frame)
				else:
					empty_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
					frame = cv2.putText(empty_frame,'RECIEVED BAD FRAME',(10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
					self.display_queue.put(frame)

				if self.state_flag.value == 3:
					break

		self.capture.release()
		self.state_flag.value = 4

		

	def display_frames(self):
		while True:
			if not self.display_queue.empty():
				frame = self.display_queue.get()
				if self.state_flag.value == 1:
					frame = cv2.putText(frame,'Recording',(10,130), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
				cv2.imshow("Video", frame)
				cv2.waitKey(1)
			
			if self.state_flag.value == 4:
				break

		while not self.display_queue.empty():
			frame = self.display_queue.get()
			cv2.imshow("Video", frame)
			cv2.waitKey(1)
		cv2.destroyAllWindows() 


	def record_frames(self,active_file):
		current_file = active_file.value
		out = cv2.VideoWriter(current_file+'.mp4', self.fourcc, self.fps, (self.width, self.height))
		print('starting up')
		while True:
			if not self.record_queue.empty():
				record_frame = self.record_queue.get()
				if record_frame is not None: 
					out.write(record_frame)
				else:
					ic('empty frame recieved while recording')
			
			if self.close_video.value == True:
				while not self.record_queue.empty():
					record_frame = self.record_queue.get()
					if record_frame is not None: 
						out.write(record_frame)
						ic('dumping last frames')
					else:
						ic('empty frame recieved while closing')

				out.release()
				if current_file != active_file.value:
					print('i know it needs to be renamed')
					os.rename(current_file+'.mp4', active_file.value+'.mp4')
				
				active_file.value = f'output{time.time()}'
				current_file = active_file.value
					
				out = cv2.VideoWriter(active_file.value+'.mp4', self.fourcc, self.fps, (self.width, self.height))
				self.close_video.value = False
				self.state_flag.value = 0

			
			if self.state_flag.value == 3:
				break

			time.sleep(.01)


class StateServer:
	def __init__(self):
		self.state = 'idle'
		manager = Manager()
		self.active_file = manager.Value(c_char_p,  f'output{time.time()}')

		self.video_display = VideoDisplay(video_source=0)
		self.capture_process = Process(target=self.video_display.capture_frames)
		self.display_process = Process(target=self.video_display.display_frames)
		self.record_process = Process(target=self.video_display.record_frames,args=(self.active_file,))


		self.capture_process.start()
		self.display_process.start()
		self.record_process.start()

	@cherrypy.expose
	def update(self, message=None):
		"""Updates the server state based on the message."""
		if message == "record":
			self.state = "started"
			self.video_display.state_flag.value = 1
			return "Server state changed to 'started'"
		
		if message == "stop_record":
			self.state = "started"
			self.video_display.state_flag.value = 2
			print('\n\n\n RECEIVED NEW STOP COMMAND \n\n\n')
			return "Server state changed to 'started'"
		
		if message == 'wig_on':
			self.video_display.wiggleometer_flag.value = 1
		if message == 'wig_off':
			self.video_display.wiggleometer_flag.value = 0
		
		elif message == "close":
			self.state = "stopped"
			self.video_display.state_flag.value = 3

			self.display_process.join()
			ic('display process is dead')
			self.capture_process.join()
			ic('capture process is dead')
			self.record_process.join()
			ic('record prcoess is dead')
			
			cherrypy.engine.exit()

			
			return "Server state changed to 'stopped'"
		else:
			return "Unknown message. Use 'start' or 'stop' to change state."
		
	@cherrypy.expose
	def name(self, message=None):
		"""Updates the server state based on the message."""
		self.active_file.value = message
		# if message:
		# 	self.active_file.value = message
		# 	pass
			


	@cherrypy.expose
	def status(self):
		"""Returns the current state of the server."""
		return f"Current server state: {self.state}"




# Usage example
if __name__ == "__main__":

	cherrypy.config.update({
		'server.socket_host': '0.0.0.0',
		'server.socket_port': 8080
	})
	cherrypy.quickstart(StateServer())





