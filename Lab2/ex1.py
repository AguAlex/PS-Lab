import numpy as np
import matplotlib.pyplot as plt

A = 1
f = 1
phi_for_sin = 1
t = 1
n = 1000
nts = np.linspace(0, t, n)

sin_signal = A * np.sin(2 * np.pi * f * nts + phi_for_sin)

# cos(x) = sin(x + pi/2)
phi_for_cos = phi_for_sin - np.pi / 2

cos_signal = A * np.cos(2 * np.pi * f * nts + phi_for_cos)

fig, axs = plt.subplots(2)
fig.suptitle("Sin vs Cos")

axs[0].plot(nts, sin_signal)
axs[1].plot(nts, cos_signal)

plt.show()
