import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter1d
import time



def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

points = []
# Load the image in grayscale
for i in range (1,26):
    image_path = f'.\Thresh_Images\Mester ({i}).png'  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    max_intense = np.max(image)
    min_intense = 1

    # Calculate the histogram
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    smooth_histogram = moving_average(histogram[:],10)
    # Fit a polynomial
    coeffs = np.polyfit(np.arange(0,len(histogram)), histogram, deg=2)
    polynomial = np.poly1d(coeffs)
    fitted_data = polynomial(np.arange(0,len(histogram)),)
    weiner_data = wiener(histogram)
    gaus_data = gaussian_filter1d(histogram,sigma =2)
    gaus_data_der = np.gradient(np.gradient(gaus_data))

    left_box_vals = image[:,0:200]
    right_box_vals = image[:,1720:1920]
    box_vals = np.append(left_box_vals,right_box_vals)
    edge_total = np.sum(box_vals)

    edge_hist = cv2.calcHist([box_vals], [0], None, [256], [0, 256]).flatten()

    start = time.time()
    radius = 20
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
    # Create a mask for the circle with the given radius
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (x - maxLoc[0])**2 + (y - maxLoc[1])**2 <= radius**2

    index = np.where(histogram < 1000)[0][0]
    average_intensity = image[mask].mean()
    cv2.circle(image, maxLoc, radius, (255,0.0), 2)
    print(maxVal,maxLoc,average_intensity)

    print(histogram)
    print(index)
    cv2.imshow('frame', image)
    plt.plot(histogram[index:])
    plt.show()

    # Calculate the average intensity within the radius
    print(f'the calc took {time.time()-start} seconds')
    print(maxVal,maxLoc,average_intensity)

    #plt.plot(gaus_data_der)
    print(np.argmax(gaus_data_der))
    print(np.sum(image)/1000000)
    print(histogram[0]/100000)
    # x = average_intensity
    # y = (histogram[0] + histogram[1])/100000
    # y = edge_total
    # points.append([x,y])
    # plt.plot(x,y,'*',label=f'{image_path}')
    # plt.plot(histogram, label='Noisy Data')
    #plt.plot(gaus_data, label='Fitted Gaus', color='red')
    #plt.legend()
    #plt.show()

    # Plot the histogram
    #plt.plot(edge_hist[1:], label='Noisy Data')
    print(edge_hist[0:10])
    #plt.plot(smooth_histogram, label='Moving Average', color='red')
    #plt.legend()
    #plt.show()
    # print(image_path)
    
plt.legend()
plt.show()
