import cv2
import numpy as np

def nothing(x):
    pass

def main():
    # Load the image
    image_path = 'ball1.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image not found.")
        return

    # Create a window
    cv2.namedWindow('Canny Edge Detection')

    # Create trackbars for threshold adjustment
    cv2.createTrackbar('Threshold1', 'Canny Edge Detection', 50, 500, nothing)
    cv2.createTrackbar('Threshold2', 'Canny Edge Detection', 150, 500, nothing)

    while True:
        # Get current positions of the trackbars
        threshold1 = cv2.getTrackbarPos('Threshold1', 'Canny Edge Detection')
        threshold2 = cv2.getTrackbarPos('Threshold2', 'Canny Edge Detection')

        # Apply Canny edge detection
        edges = cv2.Canny(image, threshold1, threshold2)

        # Display the edge-detected image
        cv2.imshow('Canny Edge Detection', edges)

        # Break the loop if the user presses the ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()