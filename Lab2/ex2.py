import random
import numpy as np
import matplotlib.pyplot as plt

def sin_signal(A, f, phi):
    return A * np.sin(2 * np.pi * f * nts + phi)

A = 1
t = 0.1
n = 10000
nts = np.linspace(0, t, n)
f = 200

original_signals = [
    sin_signal(A, f, 1),
    sin_signal(A, f, 0.5),
    sin_signal(A, f, 2),
    sin_signal(A, f, 4),
]

fig, axs = plt.subplots(4)
fig.suptitle("Original Sin Signals")

for x in range(4):
    axs[x].plot(nts, original_signals[x])

plt.show()

# vector de zgomot aleator cu medie 0 si varianta 1
z = np.random.normal(size=(1, n))[0]

SNRs = [0.1, 1, 10, 100]
SNR_signals = []

for SNR in SNRs:

    x_norm = np.linalg.norm(original_signals[0])
    z_norm = np.linalg.norm(z)

    gamma = x_norm / (z_norm * np.sqrt(SNR)) 
    SNR_signals.append(original_signals[0] + gamma * z) # aplicare zgomot

fig, axs = plt.subplots(4)
fig.suptitle("Sinus + Zgomot")

for i, snr_val in enumerate(SNRs):
    axs[i].plot(nts, SNR_signals[i])
    axs[i].set_ylabel(f"SNR={snr_val}")

plt.show()