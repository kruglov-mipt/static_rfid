import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 110, 10)

y1 = np.array([0, 25, 60, 70, 105, 145, 170, 200, 230, 280, 330])

y2 = np.array([0, 22, 53, 60, 92, 133, 156, 185, 215, 263, 304])

print(x)




lines = plt.plot(x, y1, x, y2)

plt.xlabel("Количество меток")

plt.ylabel("Количество слотов")

plt.legend(['Fast Q', 'Adaptive Q' ])

plt.grid()

plt.show()