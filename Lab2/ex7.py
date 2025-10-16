import matplotlib.pyplot as plt
import numpy as np

def sin_signal(A, f, phi, nts):
    return A * np.sin(2 * np.pi * f * nts + phi)

t = 1
n = 1000
nts = np.linspace(0, t, n)
nts_4 = np.linspace(0, t, int(n / 4))
nts_16 = np.linspace(0, t, int(n / 16))

f_signal = sin_signal(1, 200, 0, nts)
decimated_4 = sin_signal(1, 200, 0, nts_4)
decimated_16 = sin_signal(1, 200, 0, nts_16)

fig, axs = plt.subplots(3)
fig.suptitle("Decimare de frecventa")

for ax in axs.flat:
    ax.set_xlim([0, 0.1])

axs[0].plot(nts, f_signal)
axs[1].plot(nts_4, decimated_4)
axs[2].plot(nts_16, decimated_16)

plt.show()

# Amplitudinea ramane constanta, dar forma si frecventa sunt distorsionate.