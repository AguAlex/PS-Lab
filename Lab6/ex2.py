import random
import matplotlib.pyplot as plt
import numpy as np

seed = 0
np.random.seed(seed)
random.seed(seed)

x = np.random.rand(100)

x2 = np.convolve(x, x)
x4 = np.convolve(x2, x2)
x8 = np.convolve(x4, x4)

fig, axs = plt.subplots(4)
fig.suptitle("Convolution")

axs[0].plot(np.linspace(1, 0, len(x)), x)
axs[1].plot(np.linspace(1, 0, len(x2)), x2)
axs[2].plot(np.linspace(1, 0, len(x4)), x4)
axs[3].plot(np.linspace(1, 0, len(x8)), x8)

plt.show()

# Semnal bloc rectangular

N = 100
rect = np.zeros(N)
rect[45:55] = 1

rect2 = np.convolve(rect, rect)
rect4 = np.convolve(rect2, rect2)
rect8 = np.convolve(rect4, rect4)

fig, axs = plt.subplots(4, figsize=(8, 10))
fig.suptitle("Convolutie semnal bloc rectangular")

axs[0].plot(rect)
axs[0].set_title("x")

axs[1].plot(rect2)
axs[1].set_title("x * x")

axs[2].plot(rect4)
axs[2].set_title("(x * x) * (x * x)")

axs[3].plot(rect8)

plt.tight_layout()
plt.show()