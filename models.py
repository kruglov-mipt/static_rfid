import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 50, 100, 200, 400, 800, 1600, 3200])

y1 = np.array([0, 0.115, 0.226, 0.465, 0.943, 1.843, 3.685, 7.41])

y2 = np.array([0, 0.122, 0.249, 0.522, 1.078, 2.137, 4.22, 8.58])

y3 = np.array([0, 0.122, 0.236, 0.485, 0.981, 1.98, 3.91, 7.692])

lines = plt.plot(x, y1, x, y2, x, y3)

plt.xlabel("Количество меток")

plt.ylabel("Время оценки, с")

plt.legend(['Pefsa', 'Adaptive Q', 'Subep-q'])

plt.grid()

plt.show()