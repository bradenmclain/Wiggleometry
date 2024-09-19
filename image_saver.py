import cv2

# Open a video file (or 0 for webcam)
video_path = '/home/delta/Wiggleometry/data/Wiggleometer_Deposits/wiggleometer_deposit_1.mp4'  # Replace with your video file path or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set a counter for saved images
img_counter = 0

# Loop to read and display video frames
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Reached the end of the video or failed to grab frame.")
        break

    # Display the current frame
    cv2.imshow("Video", frame)

    # Wait for a key event for 1ms
    key = cv2.waitKey(60) & 0xFF

    # Press spacebar to save the current frame as an image
    if key == ord(' '):
        img_name = f"saved_frame_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        img_counter += 1

    # Press 'q' to exit the video
    elif key == ord('q'):
        print("Exiting...")
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()