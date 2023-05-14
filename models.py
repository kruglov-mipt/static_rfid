import numpy as np
import matplotlib.pyplot as plt

x = np.array([50, 100, 200, 400, 800, 1600, 3200])

y1 = np.array([0.122, 0.232, 0.475, 0.951, 1.91, 3.81, 7.622])

y2 = np.array([0.118, 0.234, 0.460, 0.935, 1.86, 3.70, 7.448])

y3 = np.array([0.132, 0.253, 0.503, 1.091, 2.186, 3.96, 7.81])

y4 = np.array([0.124, 0.255, 0.526, 1.062, 2.074, 4.29, 8.58])

lines = plt.plot(x, y1, x, y2, x, y3, x, y4)

plt.xlabel("Количество меток")

plt.ylabel("Время, с")

plt.legend(['Subep-q', 'Tafsa-1', 'Chen', 'Fast-q' ])

plt.grid()

plt.show()