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
        self.width, self.height, rgb= self.frame.shape
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.queue = queue

        print(self.height,self.width,self.fps)


    def get_frame(self):
        self.state, self.frame = self.cap.read()


    def fill_video_queue(self):
        self.queue.put(np.copy(self.frame))
        print('frame def put into queue')


def save_video_queue(image_queue,flag_queue,height,width,fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'output{time.time()}.avi', fourcc, fps, (width, height))
    flag = 'idle'
    
    while flag != 'close':
        ic(flag)
        
        if flag_queue.empty() == False:
            flag = flag_queue.get()
        
        if flag == 'recording':
            while image_queue.empty() == False:
                frame = image_queue.get()
                if frame is not None: 
                    ic(frame)
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
            name = time.time()
            out = cv2.VideoWriter(f'output{name}.avi', fourcc, fps, (width, height))

        time.sleep(.1)

    
    out.release()
    try:
        os.remove(f'output{name}.avi')
    except:
        pass
    ic('closing')


if __name__ == '__main__':
    cv2.namedWindow("preview",cv2.WND_PROP_FULLSCREEN)
    cv2.resizeWindow("preview", 960, 540) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    record = False
    flag_queue = Queue()
    image_queue = Queue()
    flag = 'idle'
    key = ''
    

    stream = VideoWidget(image_queue)   
    
    if stream.cap.isOpened(): # try to get the first frame
        stream.get_frame()
        process = Process(target=save_video_queue, args=(image_queue, flag_queue,stream.width,stream.height,stream.fps))
        process.start()

    else:
        stream.state = False

    while stream.state:               
        if key == 27: # exit on ESC
            record = False
            if flag == 'recording':
                flag = 'stop_recording'
                flag_queue.put(flag)

            flag = 'close'
            flag_queue.put(flag)
            stream.cap.release()
            stream.state = False

        if key == ord('r') or record:
            record = True
            if flag != 'recording':
                flag = "recording"
                flag_queue.put(flag)
            flag = 'recording'
            start = time.time()
            print('frame put into recording')
            stream.fill_video_queue()
            print(f'recording took {time.time()-start} seconds')
            
        if key == ord('s'):
            record = False
            flag = "stop_recording"
            flag_queue.put(flag)
            flag = 'idle'
            flag_queue.put(flag)

        display_frame = cv2.putText(stream.frame,flag,(10,140), font, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("preview", stream.frame)
        key = cv2.waitKey(1)
        if flag != 'close':
            stream.get_frame()


    process.join()
    process.close()
    cv2.destroyWindow("preview")
    stream.cap.release()