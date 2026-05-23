import numpy as np
import matplotlib.pyplot as plt

def f(n, x):
    return n/np.sqrt(np.pi) * np.exp(-n**2 * x**2)

x = np.linspace(-10, 10, 10000)
for i in range(1, 8):
    plt.plot(x, f(i, x), label=f"f_{i}")
plt.legend(loc="best")
plt.show()