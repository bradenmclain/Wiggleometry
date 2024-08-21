import cv2
import numpy as np

def threshold_image_by_rgb(image, lower_rgb, upper_rgb):
    # Apply the threshold using the given RGB ranges
    lower_bound = np.array(lower_rgb, dtype="uint8")
    upper_bound = np.array(upper_rgb, dtype="uint8")
    mask = cv2.inRange(image, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def nothing(x):
    pass

def main():
    # Load the image
    image_path = 'ball1.png'
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found.")
        return

    # Create a window
    cv2.namedWindow('Thresholded Image')

    # Create trackbars for RGB threshold adjustment
    cv2.createTrackbar('R_min', 'Thresholded Image', 0, 255, nothing)
    cv2.createTrackbar('R_max', 'Thresholded Image', 255, 255, nothing)
    cv2.createTrackbar('G_min', 'Thresholded Image', 0, 255, nothing)
    cv2.createTrackbar('G_max', 'Thresholded Image', 255, 255, nothing)
    cv2.createTrackbar('B_min', 'Thresholded Image', 0, 255, nothing)
    cv2.createTrackbar('B_max', 'Thresholded Image', 255, 255, nothing)

    while True:
        # Get current positions of the trackbars
        r_min = cv2.getTrackbarPos('R_min', 'Thresholded Image')
        r_max = cv2.getTrackbarPos('R_max', 'Thresholded Image')
        g_min = cv2.getTrackbarPos('G_min', 'Thresholded Image')
        g_max = cv2.getTrackbarPos('G_max', 'Thresholded Image')
        b_min = cv2.getTrackbarPos('B_min', 'Thresholded Image')
        b_max = cv2.getTrackbarPos('B_max', 'Thresholded Image')

        # Set the lower and upper RGB boundaries based on trackbar positions
        lower_rgb = [b_min, g_min, r_min]
        upper_rgb = [b_max, g_max, r_max]

        # Apply the threshold to the image
        thresholded_image = threshold_image_by_rgb(image, lower_rgb, upper_rgb)

        # Display the thresholded image
        cv2.imshow('Thresholded Image', thresholded_image)

        # Break the loop if the user presses the ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()