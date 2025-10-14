import numpy as np
import matplotlib.pyplot as plt

# A, B, C, D

def a(t):
    return np.sin(800 * np.pi * t)

t1 = np.linspace(0, 0.1, 1600)

def b(t):
    return np.sin(1600 * np.pi * t)

t2 = np.linspace(0, 3, 2400) # 3 * 800Hz

def c(t):
    return 2 * (t * 240 - np.floor(0.5 + t * 240))

def d(t):
    return np.sign(np.sin(2 * np.pi * 300 * t))

t3 = np.linspace(0, 0.02, 1000)

fig, axs = plt.subplots(4)
axs[0].plot(t1, a(t1))
axs[1].plot(t2, b(t2))
axs[2].plot(t3, c(t3))
axs[3].plot(t3, d(t3))
plt.show()

# D

I_e = np.random.rand(128, 128)

plt.imshow(I_e, cmap='gray')

plt.show()

# E

I_f = np.zeros((128,128))

# Coloanele devin i / 127 => gradient
for i in range(128):
    I_f[:, i] = i / 127

I_f[40:90, 40:90] = 1  

plt.imshow(I_f, cmap='gray')

plt.show()
