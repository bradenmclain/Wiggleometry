import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

# Sample data for binary_change with a range for demonstration
binary_change = np.linspace(45000, 1700000, 500)  # Example range for demonstration

# Define deposit data for indexing
deposit_data = {'deposit_start_idx': 0, 'deposit_end_idx': len(binary_change)}

# Get the colormap (OrRd)
cmap = plt.get_cmap('OrRd')

# Normalize the binary_change values
norm = Normalize(vmin=45000, vmax=1700000)  # Use the same range for normalization

# Create a figure and axis
fig, ax = plt.subplots()

# Plot bead-like segments using circular markers
num_beads = len(binary_change[deposit_data['deposit_start_idx']:deposit_data['deposit_end_idx']])
x_positions = np.arange(num_beads)  # x positions based on index

for i in range(num_beads):
    # Normalize the value to get a color from the colormap
    color = cmap(norm(binary_change[i + deposit_data['deposit_start_idx']]))
    
    # Plot each bead as a circular marker with specific color
    ax.plot(binary_change[i], 0, 'o', color=color, markersize=10)  # Directly using the index for x

# Set limits for the plot to fit all beads
ax.set_xlim(-1, np.max(binary_change)*1.02)  # x limits to fit all beads
ax.set_ylim(-0.5, 0.5)  # Bring the y-limits closer together

# Define desired min and max tick values
min_tick_value = 45000
max_tick_value = 1700000

# Create tick positions based on the desired range
num_ticks = 5
tick_positions = np.linspace(min_tick_value, max_tick_value, num_ticks)  # Use your actual range

# Set tick marks and labels
ax.set_xticks(tick_positions)

# Create a formatter function for scientific notation
def scientific(x, pos):
    return f'{x:.1e}'  # Format to scientific notation

# Set the formatter for the x-axis
ax.xaxis.set_major_formatter(FuncFormatter(scientific))

# Move ticks closer to the beads by adjusting their vertical position
ax.tick_params(axis='x', pad=5)  # Decrease padding to move ticks closer to the line

# Adjust the axis appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(('outward', 10))  # Move the bottom spine down slightly

# Remove the y-axis
ax.get_yaxis().set_visible(False)

# Display the plot
plt.show()
