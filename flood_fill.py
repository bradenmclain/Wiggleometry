import cv2
import numpy as np
import time

image_path = f'.\Thresh_Images\Mester ({1}).png'  # Replace with your image path
im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

h,w = im.shape
start = time.time()
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(im)
brightest_pixel = max_loc  # (x, y) coordinates of the brightest pixel
#brightest_pixel = (max_loc[1],max_loc[0])

mask = np.zeros((h+2,w+2),np.uint8)

floodflags = 4
floodflags |= cv2.FLOODFILL_MASK_ONLY
floodflags |= (255 << 8)

num,im,mask,rect = cv2.floodFill(im, mask, brightest_pixel, (255), (2), (2), floodflags)
print(f'it took {time.time()-start}')
cv2.imshow("frame", mask)
cv2.waitKey(0)