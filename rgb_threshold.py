import cv2
import numpy as np
import matplotlib.pyplot as plt

def threshold_image_by_rgb(frame, lower_rgb, upper_rgb):
    # Apply the threshold using the given RGB ranges
    lower_bound = np.array(lower_rgb, dtype="uint8")
    upper_bound = np.array(upper_rgb, dtype="uint8")
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result, mask

def count_pixels(mask):
    # Count the number of pixels above and below the threshold
    above_threshold = np.count_nonzero(mask)
    total_pixels = mask.size
    below_threshold = total_pixels - above_threshold
    return above_threshold, below_threshold

def nothing(x):
    pass

def main():
    # Load the video
    video_path = '/home/delta/Wiggleometry/data/Second/wiggleometer_deposit_2.mp4'
    cap = cv2.VideoCapture(video_path)
    #plt.ion()
    if not cap.isOpened():
        print("Error: Video not found or cannot be opened.")
        return

    # Create a window
    cv2.namedWindow('Thresholded Video')

    # Create trackbars for RGB threshold adjustment
    cv2.createTrackbar('R_min', 'Thresholded Video', 0, 255, nothing)
    cv2.createTrackbar('R_max', 'Thresholded Video', 255, 255, nothing)
    cv2.createTrackbar('G_min', 'Thresholded Video', 0, 255, nothing)
    cv2.createTrackbar('G_max', 'Thresholded Video', 255, 255, nothing)
    cv2.createTrackbar('B_min', 'Thresholded Video', 0, 255, nothing)
    cv2.createTrackbar('B_max', 'Thresholded Video', 255, 255, nothing)
    above = []
    plt.ion()
    plt.xlim([0,800])
    plt.ylim([0,100000])
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("End of video reached or error occurred.")
            break
        


        # Get current positions of the trackbars
        r_min = cv2.getTrackbarPos('R_min', 'Thresholded Video')
        r_max = cv2.getTrackbarPos('R_max', 'Thresholded Video')
        g_min = cv2.getTrackbarPos('G_min', 'Thresholded Video')
        g_max = cv2.getTrackbarPos('G_max', 'Thresholded Video')
        b_min = cv2.getTrackbarPos('B_min', 'Thresholded Video')
        b_max = cv2.getTrackbarPos('B_max', 'Thresholded Video')

        r_min = 255
        r_max = 255
        g_min = 255
        g_max = 255
        b_min = 100
        b_max = 254

        # Set the lower and upper RGB boundaries based on trackbar positions
        lower_rgb = [b_min, g_min, r_min]
        upper_rgb = [b_max, g_max, r_max]

        # Apply the threshold to the current frame
        thresholded_frame, mask = threshold_image_by_rgb(frame, lower_rgb, upper_rgb)

        # Count the number of pixels above and below the threshold
        above_threshold, below_threshold = count_pixels(mask)

        # Display the counts
        above.append(above_threshold)
        plt.clf()
        plt.plot(above)
        plt.draw()
        plt.pause(.01)

        # Display the thresholded frame
        cv2.imshow('Thresholded Video', frame)

        # Break the loop if the user presses the ESC key
        if cv2.waitKey(5) & 0xFF == 27:
            break

        #plt.draw()
    plt.ioff()

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()
    plt.plot(above)
    plt.show()

if __name__ == "__main__":
    main()