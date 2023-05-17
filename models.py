import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 50, 100, 200, 400, 800, 1600, 3200])

y1 = np.array([0, 162, 320, 730, 1480, 3060, 6300, 12800])

y2 = np.array([0, 239, 424, 750, 1450, 2800, 5800, 11200])



lines = plt.plot(x, y1, x, y2, )

plt.xlabel("Количество меток")

plt.ylabel("Количество слотов")

plt.legend(['Adaptive Q', 'Chen algorithm'])

plt.grid()

plt.show()