import numpy as np

x = 10 * np.random.rand(50)
y = 2 * x - 1 + np.random.rand(50)

import matplotlib.pyplot as plt

plt.plot(x, y)
plt.show()