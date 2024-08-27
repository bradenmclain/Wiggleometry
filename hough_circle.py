import cv2
import numpy as np

def nothing(x):
    pass

def main():
    # Load the video
    video_path = '/home/delta/Wiggleometry/data/Second/wiggleometer_deposit_2.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Video not found or cannot be opened.")
        return

    # Create a window
    cv2.namedWindow('Hough Circle Transform')

    # Create trackbars for adjusting Hough Circle parameters
    cv2.createTrackbar('dp', 'Hough Circle Transform', 1, 10, nothing)
    cv2.createTrackbar('minDist', 'Hough Circle Transform', 20, 100, nothing)
    cv2.createTrackbar('param1', 'Hough Circle Transform', 50, 300, nothing)
    cv2.createTrackbar('param2', 'Hough Circle Transform', 30, 300, nothing)
    cv2.createTrackbar('minRadius', 'Hough Circle Transform', 0, 100, nothing)
    cv2.createTrackbar('maxRadius', 'Hough Circle Transform', 0, 1000, nothing)

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("End of video reached or error occurred.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to reduce noise and improve circle detection
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Get current positions of the trackbars
        dp = cv2.getTrackbarPos('dp', 'Hough Circle Transform')
        minDist = cv2.getTrackbarPos('minDist', 'Hough Circle Transform')
        param1 = cv2.getTrackbarPos('param1', 'Hough Circle Transform')
        param2 = cv2.getTrackbarPos('param2', 'Hough Circle Transform')
        minRadius = cv2.getTrackbarPos('minRadius', 'Hough Circle Transform')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'Hough Circle Transform')

        # Ensure dp is at least 1 to avoid division by zero
        dp = max(dp, 1)

        # Perform Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
            param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius
        )

        # Create a copy of the original frame to draw circles on
        output_frame = frame.copy()

        # If some circles are detected, draw them
        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                # Draw the outer circle
                cv2.circle(output_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(output_frame, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Display the resulting frame with detected circles
        cv2.imshow('Hough Circle Transform', output_frame)

        # Break the loop if the user presses the ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()