import imageio
import matplotlib.pyplot as plt
import time

# Path to the video file
video_path = '/home/delta/Downloads/output_video.mp3'

import pygame

# Initialize pygame
pygame.init()

# Path to the video file
video_path = '/home/delta/Downloads/TSSF_02_04_R017_L14T7.avi'

# Read the video file
reader = imageio.get_reader(video_path)
fps = reader.get_meta_data()['fps']

# Get the size of the video frames
video_size = reader.get_meta_data()['size']

# Create a pygame window with the same size as the video
screen = pygame.display.set_mode(video_size)
pygame.display.set_caption('Video Playback')

# Clock to control frame rate
clock = pygame.time.Clock()

# Play video frame by frame
for i, frame in enumerate(reader):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
    print(i)

    # Convert frame (numpy array) to a surface for pygame
    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))  # Swap axes to match Pygame format

    # Display the frame
    screen.blit(frame_surface, (0, 0))
    pygame.display.update()

    # Control playback speed to match the video's FPS
    clock.tick(fps)

# Clean up
reader.close()
pygame.quit()
