import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import sounddevice

t = 1
n = 1000
nts = np.linspace(0, t, n)

sin_low = np.sin(400 * np.pi * nts)
sin_high = np.sin(2000 * np.pi * nts)
low_high_concat = np.concatenate((sin_low, sin_high))

# plt.plot(nts, low_high_concat)
# plt.show()

wav.write(r"C:\Users\Agu\Desktop\low_vs_high.wav", 44100, low_high_concat)