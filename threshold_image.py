import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load the grayscale image
for i in range(1,25):
    image_path = f'./Thresh_Images/mester ({i}).png'  # Update with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    # Create the figure and the axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    threshold = 0  # Initial threshold value

    # Apply the initial binary threshold
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    img_display = ax.imshow(binary_image, cmap='gray')

    # Add a slider for threshold
    ax_threshold = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    threshold_slider = Slider(ax_threshold, 'Threshold', 0, 255, valinit=threshold, valstep=1)

    # Update function for the slider
    def update(val):
        threshold = threshold_slider.val
        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        img_display.set_data(binary_image)
        fig.canvas.draw_idle()

    threshold_slider.on_changed(update)

    plt.show()
    print(threshold_slider.val)
    