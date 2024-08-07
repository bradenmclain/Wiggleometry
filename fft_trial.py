import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Load the image in grayscale
image_path = f'.\Thresh_Images\Binary_2.png' # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

_, binary_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

cv2.imshow('frame',binary_image)
cv2.waitKey(0)
f_transform = fft2(binary_image)
f_transform_shifted = fftshift(f_transform)

# Analyze the magnitude of the frequency components
magnitude_spectrum = np.abs(f_transform_shifted)
plt.imshow(np.log(magnitude_spectrum + 1), cmap='gray')
plt.title('Frequency Spectrum')
plt.colorbar()
plt.show()

# Optionally apply some frequency domain filtering
# Example: low-pass filter or high-pass filter

# Apply Inverse FFT to get the filtered image
filtered_f_transform_shifted = ifftshift(f_transform_shifted)
filtered_image = ifft2(filtered_f_transform_shifted).real

plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.show()