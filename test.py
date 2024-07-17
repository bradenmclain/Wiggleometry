import matplotlib.pyplot as plt
import numpy as np
import time
from draggable_lines import draggable_lines



x = np.linspace(1,2000,500)
y = np.linspace(1,500,500)



fig = plt.figure()
ax = fig.add_subplot(111)
plt.subplot(111)
plt.plot(x,y,label = 'data')
plt.legend(loc='upper right')
Vline = draggable_lines(ax, "h", 500,np.max(x))
# Update the legend after adding the draggable line
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper right')


while Vline.XorY != 0:

    print(f'the position is {Vline.XorY}')
    plt.draw()
    plt.pause(.1)

plt.show()