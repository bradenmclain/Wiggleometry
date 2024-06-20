import cv2
import time
import icecream as ic
from multiprocessing import Process


def threaded_print(thing):
    print(thing)


if __name__ == '__main__':

    cv2.namedWindow("preview",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("preview", 960, 540) 
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    font = cv2.FONT_HERSHEY_SIMPLEX

    proc = Process(target=threaded_print)  # instantiating without any argument


    
    if cap.isOpened(): # try to get the first frame
        rval, frame = cap.read()
        recording_flag = "PRESS R TO START RECORDING"
        
    else:
        rval = False

    while rval:
        key = cv2.waitKey(1)
        start2 = time.time()
        rval, frame = cap.read()
        if key == 27: # exit on ESC
            break

        if key == ord('r'):
            recording_flag = "PRESS S TO STOP RECORDING"
            

        if key == ord('s'):
            recording_flag = "PRESS R TO START RECORDING"
            proc = Process(target=threaded_print, args=(recording_flag,))
            proc.start()
            print('hitting')

        display_frame = cv2.putText(frame,recording_flag,(10,140), font, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("preview", display_frame)

    cv2.destroyWindow("preview")
    cap.release()