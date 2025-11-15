
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import butter, cheby1, lfilter

train_df = pd.read_csv("Lab6/data/Train.csv", parse_dates=["Datetime"], dayfirst=True)
train_samples = train_df["ID"].values
train_signal = train_df["Count"].values
train_sample_rate = 1 / 3600

# A

size = 72 # 3 zile * 24 esantioane
x = train_signal[:size]
x_samples = train_samples[:size]
ws = [5, 9, 13, 17]

# B

filtered_signals = []
for w in ws:
    filtered_signals.append(np.convolve(x, np.ones(w), "valid") / w)

fig, axs = plt.subplots(4, 1, figsize=(8, 10))
fig.suptitle("Filtru de tip medie")

for i, (w, filtered_signal) in enumerate(zip(ws, filtered_signals)):
    axs[i].plot(
        np.linspace(0, size, len(filtered_signal)),
        filtered_signal
    )
    axs[i].set_title(f"window_size = {w}")
    axs[i].grid(True)

fig.tight_layout()
plt.show()
