import matplotlib.pyplot as plt
import numpy as np

def rectangle_window(Nw):
    return np.array([1 for i in range(Nw)])

def hanning_window(Nw):
    return np.array([0.5 * (1 - np.cos(2 * np.pi * i / Nw)) for i in range(Nw)])

N = 1000
Nw = 200
t = 1
nts = np.linspace(0, t, N)

signal = 1 * np.sin(2 * np.pi * 100 * nts + 0)

# Padding cu valori de 0
rectangle = np.pad(rectangle_window(Nw), pad_width=(0, N - Nw), mode="constant", constant_values=0)
hanning = np.pad(hanning_window(Nw), pad_width=(0, N - Nw), mode="constant", constant_values=0)

rectangle_signal = signal * rectangle
hanning_signal = signal * hanning

plot_name = "Windows"
fig, axs = plt.subplots(2)
plt.suptitle(plot_name)

axs[0].plot(nts, rectangle_signal)
axs[0].set_title("Rectangle")

axs[1].plot(nts, hanning_signal)
axs[1].set_title("Hanning")

plt.tight_layout()
plt.show()