import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global variables for bounding box coordinates
start_point = None
end_point = None
drawing = False

# Mouse callback function to draw the rectangle
def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = start_point

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

# Function to calculate histograms
def calculate_histograms(image, bbox):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract bounding box coordinates
    x1, y1, x2, y2 = bbox
    inside_box = gray_image[y1:y2, x1:x2]

    # Create mask for everything outside the bounding box
    mask = np.ones(gray_image.shape, dtype=bool)
    mask[y1:y2, x1:x2] = False
    outside_box = gray_image[mask].flatten()

    # Calculate histograms
    hist_inside = cv2.calcHist([inside_box], [0], None, [255], [0, 256]).flatten()
    hist_outside = cv2.calcHist([outside_box], [0], None, [255], [0, 256]).flatten()

    return hist_inside, hist_outside

# Load the image
image_path = 'saved_frame_3.png'  # Update with your image path
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not open or find the image.")
    exit()

# Create a named window and set the mouse callback function
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_rectangle)

# Main loop to display the image and allow drawing the bounding box
while True:
    img_copy = image.copy()
    
    # Draw the rectangle if the start and end points are defined
    if start_point and end_point:
        cv2.rectangle(img_copy, start_point, end_point, (0, 255, 0), 2)
    
    # Display the image
    cv2.imshow('Image', img_copy)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ensure we have a valid bounding box before calculating histograms
if start_point and end_point:
    bbox = (min(start_point[0], end_point[0]), 
             min(start_point[1], end_point[1]), 
             max(start_point[0], end_point[0]), 
             max(start_point[1], end_point[1]))

    # Calculate histograms for the selected area
    hist_inside, hist_outside = calculate_histograms(image, bbox)

    # Create the histogram plots
    plt.figure(figsize=(10, 5))
    
    # Plot histogram for inside bounding box
    plt.subplot(1, 2, 1)
    plt.title('Inside Bounding Box Histogram')
    plt.plot(hist_inside, color='blue')
    plt.xlim(0, 255)
    
    # Plot histogram for outside bounding box
    plt.subplot(1, 2, 2)
    plt.title('Outside Bounding Box Histogram')
    plt.plot(hist_outside, color='green')
    plt.xlim(0, 255)

    plt.tight_layout()
    plt.show()

# Cleanup
cv2.destroyAllWindows()
