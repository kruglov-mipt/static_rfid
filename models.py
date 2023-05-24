import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 50, 100, 200, 400, 800, 1600, 3200])

y1 = np.array([0, 0.124, 0.235, 0.465, 0.96, 1.94, 3.82, 7.92])

y2 = np.array([0, 0.122, 0.239, 0.475, 0.94, 1.9, 3.86, 7.61])



lines = plt.plot(x, y1, x, y2, )

plt.xlabel("Количество меток")

plt.ylabel("Длительность идентификации, с")

plt.legend(['ILCM', 'Chen algorithm'])

plt.grid()

plt.show()