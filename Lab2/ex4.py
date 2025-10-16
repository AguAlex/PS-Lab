import numpy as np
import matplotlib.pyplot as plt

def sin(t):
    return np.sin(800 * np.pi * t)

def sawtooth(t):
    return 2 * (t * 240 - np.floor(0.5 + t * 240))

t = 0.05
n = 1000000
nts = np.linspace(0, t, n)

signal_sin = sin(nts)
signal_sawtooth = sawtooth(nts)

sin_sawtooth_added = np.add(signal_sin, signal_sawtooth)

fig, axs = plt.subplots(3)
axs[0].plot(nts, signal_sin)
axs[1].plot(nts, signal_sawtooth)
axs[2].plot(nts, sin_sawtooth_added)
plt.show()