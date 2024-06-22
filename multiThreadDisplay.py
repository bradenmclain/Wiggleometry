import cv2
import time
from icecream import ic
from multiprocessing import Process, Queue
import schedule
import os
import numpy as np

class VideoWidget(): 

    def __init__(self,queue):

        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  

        self.state, self.frame = self.cap.read()


        self.i = 0


        self.queue = queue


    def get_frame(self):
        self.state, self.frame = self.cap.read()


    def fill_video_queue(self):
        self.queue.put(self.frame)

    
    def close_program(self):
        os.remove(f'output{self.i}.avi')
        self.cap.release()


def save_video_queue(image_queue,flag_queue):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'output{time.time()}.avi', fourcc, 30.0, (1920, 1080))
    flag = ''
    
    while flag != 'close':
        ic(flag)
        if flag_queue.empty() == False:
            flag = flag_queue.get()
        
        if flag == 'recording':
            while image_queue.empty() == False:
                frame = image_queue.get()
                if frame is not None: 
                    out.write(frame)
                else:
                    ic('empty frame recieved while recording')

        if flag == 'stop_recording':
            ic('im stopping')
            while image_queue.empty() == False:
                frame = image_queue.get()
                if frame is not None: 
                    out.write(frame)
                else:
                    ic('empty frame recieved while finalizing recording')
            out.release()
            out = cv2.VideoWriter(f'output{time.time()}.avi', fourcc, 30.0, (1920, 1080))

        time.sleep(1)
    


if __name__ == '__main__':
    cv2.namedWindow("preview",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("preview", 960, 540) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    record = False
    flag_queue = Queue()
    image_queue = Queue()
    flag = 'idle'
    i=0
    key = 'hi'
    

    stream = VideoWidget(image_queue)
    #schedule.every(1).seconds.do(stream.save_video_queue)
    
    
    if stream.cap.isOpened(): # try to get the first frame
        stream.get_frame()
        process = Process(target=save_video_queue, args=(image_queue, flag_queue))
        process.start()        
    else:
        stream.state = False

    while stream.state:        
        start2 = time.time()
        
        if key == 27: # exit on ESC
            if flag == 'recording':
                flag == 'stop_recording'
                flag_queue.put(flag)

            stream.close_program()
            break

        if key == ord('r') or record:
            record = True
            flag = "recording"
            flag_queue.put(flag)
            start = time.time()
            stream.fill_video_queue()
            print(f'recording took {time.time()-start} seconds')
            
        if key == ord('s'):
            record = False
            flag = "stop"
            flag_queue.put(flag)
            print('hitting')

        display_frame = cv2.putText(stream.frame,flag,(10,140), font, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("preview", stream.frame)
        key = cv2.waitKey(1)
        prev_frame = stream.frame
        stream.get_frame()
        print(np.sum(stream.frame-prev_frame))
        if np.sum(stream.frame-prev_frame) ==0:
            print('SHOWING NOTHING')
        i+=1
        print(f'on frame {i}')


    cv2.destroyWindow("preview")