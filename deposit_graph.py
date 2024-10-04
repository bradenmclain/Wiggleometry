import matplotlib.pyplot as plt

data = [[1500,100],[1450,33.33],[1400,4.49],[1350,3.57],[1300,9.67],[1250,9.82],[1200,5.25],[1150,44.36],[1100,95.26]]

x_values = [point[0] for point in data]
y_values = [point[1] for point in data]


plt.ylabel('Average Time Spent Unstabe (%)')
plt.xlabel('Laser Power (W)')
plt.rcParams.update({'font.size': 18})
plt.plot(x_values,y_values)
plt.show()