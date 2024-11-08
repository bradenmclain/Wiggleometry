import cv2
import numpy as np

# Load the image
image_path = 'saved_frame_2.png'  # Update with your image path

image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not open or find the image.")
    exit()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
cv2.imshow('Grayscale Image', image)

# Define the threshold value
threshold_value = 100  # Adjust this value as needed

# Create a mask where pixels below the threshold are set to 0 (black)
mask = gray_image >= threshold_value

# Create an output image initialized to black
output_image = np.zeros_like(image)

# Apply the mask to the original colored image
output_image[mask] = image[mask]

# Display the thresholded image
cv2.imshow('Thresholded Image with Color Preservation', output_image)

# Wait indefinitely until a key is pressed
cv2.waitKey(0)

# Cleanup
cv2.destroyAllWindows()
