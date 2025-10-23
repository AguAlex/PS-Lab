import matplotlib.pyplot as plt
import numpy as np

def sin_signal(A, f, phi, nts):
    return A * np.sin(2 * np.pi * f * nts + phi)

# DFT
def fourier_transform(x):
    n = len(x)
    X = np.zeros(n, dtype=np.complex128)
    for m in range(n):
        component_sum = 0
        for k in range(n):
            component_sum += x[k] * np.exp(-2 * np.pi * 1j * k * m / n)
        X[m] = component_sum
    return X

# Calculate the winding frequency of a signal on the unit circle
def winding_frequency_on_unit_circle(x, omegas):
    z = []
    for omega in omegas:
        zw = [x[i] * np.exp(-2 * np.pi * 1j * omega * i / len(x)) for i in range(len(x))]
        z.append(zw)
    return np.array(z)

# FT at specified winding frequencies
def fourier_transform_using_winding_frequency(x, omegas):
    X = {}
    winding_frequencies = winding_frequency_on_unit_circle(x, omegas)
    for index, vector in enumerate(winding_frequencies):
        X[omegas[index]] = np.sum(vector)
    return X

n = 1000
t = 1
ts = np.linspace(0, t, n)
f = np.arange(start=0, stop=n, step=1)

signal = (
    sin_signal(1, 3, np.pi / 2, ts)
    + sin_signal(1 / 4, 7, 0, ts)
    + sin_signal(4, 15, np.pi * 3 / 4, ts)
)

omegas = [3, 7, 15]
fourier_transform_signal = fourier_transform(signal)
fourier_transform_winding_frequency = fourier_transform_using_winding_frequency(signal, omegas)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle("Fourier Transform")

# Semnalul in timp
axs[0].plot(ts, signal)
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("x(t)")

# Modulul DFT complet
axs[1].stem(f, np.abs(fourier_transform_signal), linefmt="k-", markerfmt="ko", basefmt=" ")
axs[1].set_xlabel("Frequency")
axs[1].set_ylabel("|X(w)|")
axs[1].set_xlim([0, 100])

# Modulul doar la frecventele cautate (omegas)
axs[2].stem(
    fourier_transform_winding_frequency.keys(),
    np.abs(list(fourier_transform_winding_frequency.values())),
    linefmt="k-", markerfmt="ko", basefmt=" "
)
axs[2].set_xlabel("Frequency")
axs[2].set_ylabel("|X(w)|")
axs[2].set_xlim([0, 100])

plt.tight_layout()
plt.show()
