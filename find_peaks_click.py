import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Lists to store manually selected regions
selected_rois_red = []
selected_rois_green = []

# Variable to track which set of points we are selecting (red or green)
selecting_red = True

def onclick(event):
    """Handles mouse click events."""
    global selecting_red
    
    if event.inaxes:  # Only register clicks within the plot area
        x_click = event.xdata
        y_click = event.ydata

        if selecting_red:
            selected_rois_red.append(x_click)
            print(f"Red ROI selected at x = {x_click}")
            # Display a red dot
            plt.plot(x_click, y_click, 'ro')  # Red dot for first set
        else:
            selected_rois_green.append(x_click)
            print(f"Green ROI selected at x = {x_click}")
            # Display a green dot
            plt.plot(x_click, y_click, 'go')  # Green dot for second set
        plt.draw()

def onkey(event):
    """Handles key press events."""
    global selecting_red
    
    if event.key == 'enter':  # Switch to green point selection after pressing Enter
        selecting_red = False
        print("Switched to green dot selection mode. Click to select points.")
        
def select_rois_and_find_peaks(y_data):
    """Allows the user to manually select ROIs and then runs find_peaks."""
    
    # Plot the 1D array
    fig, ax = plt.subplots()
    ax.plot(y_data, label='Data')
    plt.ylim(-1.5,1.5)
    ax.set_title('Click near where you expect peaks. Press Enter to switch to green point selection. Close the plot when done.')
    plt.legend()

    # Connect the click event to the onclick function
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Connect the key press event to switch point sets
    kid = fig.canvas.mpl_connect('key_press_event', onkey)

    # Display the plot for user input
    plt.show()

    # Disconnect the events after the plot is closed
    fig.canvas.mpl_disconnect(cid)
    fig.canvas.mpl_disconnect(kid)
    
    # Convert ROIs to integer indices (rounding to nearest index)
    roi_indices_red = [int(round(x)) for x in selected_rois_red]
    roi_indices_green = [int(round(x)) for x in selected_rois_green]
    red_peaks = []
    green_peaks = []

    for roi in roi_indices_red:
        window_size = 40
        # Define a window around each ROI (e.g., +- 10 indices)
        window_start = int(max(0, roi - (window_size/2)))
        window_end = int(min(len(y_data), roi + window_size/2))
        print(window_start)
        print(window_end)
        
        # Find peaks in the selected window
        peaks, _ = find_peaks(y_data[window_start:window_end],prominence=0.01)

        # Adjust peak indices to align with the full array
        peaks = peaks + window_start
        red_peaks.extend(peaks)
    
    for roi in roi_indices_green:
        window_size = 40
        # Define a window around each ROI (e.g., +- 10 indices)
        window_start = int(max(0, roi - (window_size/2)))
        window_end = int(min(len(y_data), roi + window_size/2))
        
        # Find peaks in the selected window
        peaks, _ = find_peaks(y_data[window_start:window_end]*-1,prominence=0.01)

        # Adjust peak indices to align with the full array
        peaks = peaks + window_start
        green_peaks.extend(peaks)
   
    # plt.plot(y_data, label='Data')
    # plt.ylim(-1.5,1.5)
    # for peak in red_peaks:
    #     plt.plot(peak, y_data[peak], 'rx', label='Red Peaks')
    # for peak in green_peaks:
    #     plt.plot(peak, y_data[peak], 'gx', label='Green Peaks')
    # plt.title('Red and Green ROI Points')
    # plt.legend()
    # plt.show()


    return red_peaks, green_peaks

# Call the function and display peaks
# roi_indices_red, roi_indices_green = select_rois_and_find_peaks(y_data)
# print(f"Red points selected at indices: {roi_indices_red}")
# print(f"Green points selected at indices: {roi_indices_green}")

# Plot the results with red and green points marked


