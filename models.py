import numpy as np
import matplotlib.pyplot as plt

x = np.array([10, 50, 100, 200, 400, 600, 800, 1000, 1400, 1800, 2500, 5000])

y1 = np.array([4, 8, 5, 2, 4, 5, 3, 2, 1.5, 1.1, 1.6, 1.5])



lines = plt.plot(x, y1,)

plt.xlabel("Количество меток")

plt.ylabel("Ошибка оценки, %")

plt.grid()

plt.show()