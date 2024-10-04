import cv2
import time
import sys

# Function to update the threshold dynamically
def update_threshold(val):
    global threshold_value
    threshold_value = val

# Open the video file
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f'video has {length} frames')

# Create a named window to display the video
cv2.namedWindow('Thresholded Video')

# Initial threshold value
threshold_value = 127

# Create a trackbar (slider) to adjust the threshold
cv2.createTrackbar('Threshold', 'Thresholded Video', threshold_value, 255, update_threshold)

# Choose which frame to select the ROI from (e.g., frame number X)
frame_X = 100  # Change this to the frame number you want to use for ROI selection

# Set the video to the desired frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_X)

# Read the frame at position X
ret, selected_frame = cap.read()
if not ret:
    print(f"Error: Unable to read frame {frame_X}.")
    cap.release()
    exit()

cap = cv2.VideoCapture(video_path)
fps    = cap.get(cv2.CAP_PROP_FPS)
print(f' fps is {fps}')

# Select ROI from the chosen frame
roi = cv2.selectROI("Select ROI", selected_frame, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Select ROI")  # Close the ROI selection window


wait_time =round((1/fps)*1000)

# Extract the ROI coordinates
x, y, w, h = roi

# Loop through the video frames
i = 0
while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or cannot fetch the frame.")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply the selected ROI to the grayscale frame
    roi_frame = gray_frame[y:y+h, x:x+w]

    # Apply the threshold on the ROI
    _, binary_frame = cv2.threshold(roi_frame, threshold_value, 255, cv2.THRESH_BINARY)

    # Show the binary image
    cv2.imshow('Thresholded Video', binary_frame)
    cv2.imshow('Original', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break
    print(f'it waited {time.time()-start}')
    print(i)
    i+=1
# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
