import cv2
from multiprocessing import Process, Value,Queue
import time

class VideoDisplay:
	def __init__(self, video_source=0):
		self.capture = cv2.VideoCapture(video_source)
		self.frame_queue = Queue(maxsize=10)  # Queue to hold frames
		self.record_queue = Queue(maxsize=60)
		self.stop_flag = Value('b', False)  # Shared flag to stop processes
		self.record_flag = Value('b', False)  # Shared flag to stop processes

	def capture_frames(self):
		fps = 25
		while True:
			start_time = time.time()
			ret, frame = self.capture.read()
			self.frame_queue.put(frame)
			if self.record_flag.value:
				self.record_queue.put(frame)
			sleep = (1/fps) - (time.time() - start_time)
			if sleep > 0:
				time.sleep(sleep)

			if self.stop_flag.value:
				break

		self.capture.release()
		print('Capture Process is free')
		

	def display_frames(self):
		while True:
			if not self.frame_queue.empty():
				frame = self.frame_queue.get()
				cv2.imshow("Video", frame)
				cv2.waitKey(10)
			
			if self.stop_flag.value:
				while not self.frame_queue.empty:
					frame = self.frame_queue.get()
					cv2.imshow("Video", frame)
					cv2.waitKey(10)
				break

		cv2.destroyAllWindows() 
		print('Display Process is free')



# Usage example
if __name__ == "__main__":
	video_display = VideoDisplay(video_source=0)
	capture_process = Process(target=video_display.capture_frames)
	display_process = Process(target=video_display.display_frames)

	capture_process.start()
	display_process.start()
	record_process.start()


	

	video_display.stop_flag.value = True

	capture_process.join()
	display_process.join()

	display_process.close()
	capture_process.close()



