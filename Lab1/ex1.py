import numpy as np
import matplotlib.pyplot as plt


# A

t1 = np.arange(0, 0.3, 0.0005)

# B) Afisare semnale

def x(t):
    return np.cos(520 * np.pi * t + np.pi // 3)

def y(t):
    return np.cos(280 * np.pi * t - np.pi // 3)

def z(t):
    return np.cos(120 * np.pi * t + np.pi // 3)

fig, axs = plt.subplots(3)
fig.suptitle("B) Grafice")
axs[0].plot(t1, x(t1))
axs[1].plot(t1, y(t1))
axs[2].plot(t1, z(t1))

plt.show()

# C) Esantionare cu 200 Hz

# t2 = np.arange(0, 0.03, 1/200) 1/200 spatiul intre esantioane
# 0.03 * 200 Hz nr de esantioane
t2 = np.linspace(0, 0.03, 6)

fig, axs = plt.subplots(3)
fig.suptitle("C) Esantionare cu 200 Hz")
axs[0].stem(t2, x(t2))
axs[0].plot(t2, x(t2))
axs[1].stem(t2, y(t2))
axs[1].plot(t2, y(t2))
axs[2].stem(t2, z(t2))
axs[2].plot(t2, z(t2))

plt.show()
