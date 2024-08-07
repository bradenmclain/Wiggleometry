import cv2
import numpy as np
import time

# Load the grayscale image
image_path = f'.\Thresh_Images\Mester ({1}).png'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

start = time.time()
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)
brightest_pixel = max_loc  # (x, y) coordinates of the brightest pixel
brightest_pixel = (max_loc[1],max_loc[0])

# Create a mask image for flood filling
h, w = image.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)  # Mask needs to be 2 pixels larger than the image

# Flood fill starting from the brightest pixel
# Arguments: image, mask, seed point, new color, lo_diff, up_diff, flags
flood_flags = 8 | cv2.FLOODFILL_MASK_ONLY  # 8-connectivity, mask only

# Set tolerance for color difference
lo_diff = 2  # Lower bound tolerance
up_diff = 4  # Upper bound tolerance

# Perform flood fill
num, image, mask, rect = cv2.floodFill(image, mask, brightest_pixel, 255, (lo_diff,)*3, (up_diff,)*3, flags=8 | cv2.FLOODFILL_MASK_ONLY)

# rect is the bounding box (x, y, width, height)
x, y, w, h = rect

# Draw the bounding box on a copy of the original image
output_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
cv2.rectangle(output_image, (x, y), (x + w, y + h), 255, 2)  # Draw the bounding box in white
print(f'it took {time.time()-start}')

# Display the image with the bounding box
cv2.imshow('Bounding Box', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()