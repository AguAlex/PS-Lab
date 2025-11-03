import matplotlib.pyplot as plt
import numpy as np

def sin_signal(A, f, phi, t):
    return A * np.sin(2 * np.pi * f * t + phi)

t = 1
n = 1000
ns = 33

x_domain = np.linspace(0, t, n)
nts = np.linspace(0, t, ns)

signals = [
    {"freq": 16, "color": "slateblue"},
    {"freq": 10, "color": "purple"},
    {"freq": 4, "color": "green"},
]

fig, axs = plt.subplots(4, 1, figsize=(8, 8))
fig.suptitle("Over-Nyquist Frequency", fontsize=14)

axs[0].plot(x_domain, sin_signal(1, 16, 0, x_domain), color="slateblue")
axs[0].set_title("16 Hz")

for ax, sig in zip(axs[1:], signals):
    y_cont = sin_signal(1, sig["freq"], 0, x_domain)
    y_samp = sin_signal(1, sig["freq"], 0, nts)

    ax.plot(x_domain, y_cont, color=sig["color"], zorder=1)
    ax.scatter(nts, y_samp, color="yellow", edgecolor="black", zorder=2)
    ax.set_title(f"{sig['freq']} Hz")

plt.tight_layout()
plt.show()
