import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import convolve

org_img = cv2.imread('balling_test.png')
# gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
# gray = gray.astype(np.float64)


# Apply identity kernel
kernel1 = np.ones((16,16))
kernel_circ = [[-1,0,0,1,1,1,0,0,-1],
               [-1,0,1,1,-2,1,1,0,-1],
               [-1,1,1,-2,-2,-2,1,1,-1],
               [-1,1,-2,-2,-2,-2,-2,1,-1],
               [1,1,-2,-2,-2,-2,-2,1,1],
               [-1,1,-2,-2,-2,-2,-2,1,-1],
               [-1,1,1,-2,-2,-2,1,1,-1],
               [-1,1,1,1,-2,1,1,1,-1],
               [-1,0,0,1,1,1,0,0,-1]]

prewitt_x = np.array([[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]])

prewitt_y = np.array([[-1, -1, -1],
         [0, 0, 0],
         [1, 1, 1]])

sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
         [-1, 0, 1]])

kernel_circ = np.asarray(kernel_circ) 
#kernel_circ[kernel_circ == .25] =3

kernel1 = kernel_circ
kernel2 = sobel_x


start = time.time()
gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float64)
cv_conv_img = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel1)
cv_conv_img = cv_conv_img * 255 / cv_conv_img.max()
cv_conv_img = cv_conv_img.astype(np.uint8)
#end = time.time()

print(f'it legit took {time.time() - start} seconds')


start = time.time()
sci_conv_img=convolve(gray,kernel1) 
sci_conv_img = sci_conv_img * 255 / (sci_conv_img.max())
sci_conv_img = sci_conv_img.astype(np.uint8)
end = time.time()
print(f'it took {end - start} seconds')


cv2.imshow(f'image',org_img)
cv2.imshow('kernel 1', cv_conv_img)
cv2.imshow('kernel 2', sci_conv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()