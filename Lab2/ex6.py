import numpy as np
import matplotlib.pyplot as plt

t = 0.01
n = 1000
nts = np.linspace(0, t, n)

sin_freq_half = np.sin(2 * n / 2 * np.pi * nts)
sin_freq_quarter = np.sin(2 * n / 4 * np.pi * nts)
sin_freq_zero = np.sin(2 * 0 * np.pi * nts)

fig, axs = plt.subplots(3)
axs[0].plot(nts, sin_freq_half)
axs[1].plot(nts, sin_freq_quarter)
axs[2].plot(nts, sin_freq_zero)
plt.show()
