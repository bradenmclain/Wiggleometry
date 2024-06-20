import cv2
import time
import icecream as ic
from multiprocessing import Process
import os


def threaded_print(thing):
    print(thing)

class VideoWidget(): 

    def __init__(self):

        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  

        self.state, self.frame = self.cap.read()


        self.i = 0

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(f'output{self.i}.avi', self.fourcc, 30.0, (1920, 1080))


    def get_frame(self):
        self.state, self.frame = self.cap.read()  

    def record_video(self):
        self.out.write(self.frame)

    def stop_recording(self):
        self.i +=1
        self.out.release()
        self.out = cv2.VideoWriter(f'output{self.i}.avi', self.fourcc, 30.0, (1920, 1080))
    
    def close_program(self):
        os.remove(f'output{self.i}.avi')
        self.cap.release()


if __name__ == '__main__':
    cv2.namedWindow("preview",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("preview", 960, 540) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    record = False
    #proc = Process(target=threaded_print)  # instantiating without any argument

    stream = VideoWidget()
    
    if stream.cap.isOpened(): # try to get the first frame
        stream.get_frame()
        recording_flag = "PRESS R TO START RECORDING"
        
    else:
        stream.state = False

    while stream.state:
        start0 = time.time()
        key = cv2.waitKey(1)
        start2 = time.time()
        
        if key == 27: # exit on ESC
            stream.close_program()
            break

        if key == ord('r') or record:
            record = True
            recording_flag = "PRESS S TO STOP RECORDING"
            start = time.time()
            stream.record_video()
            print(f'recording took {time.time()-start} seconds')
            
        if key == ord('s'):
            record = False
            recording_flag = "PRESS R TO START RECORDING"
            stream.stop_recording()
            print('hitting')


        display_frame = cv2.putText(stream.frame,recording_flag,(10,140), font, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("preview", display_frame)
        stream.get_frame()
        print(f'everything took {time.time()-start0} seconds')
        
        #proc = Process(target=threaded_print, args=(recording_flag,))
        #proc.start()

    cv2.destroyWindow("preview")