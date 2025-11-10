# A
# Semnalul este masurat din ora in ora, deci timpul de esantionare = 3600s
# Frecventa de esantiomnare = 1/Ts, Fs = 1/3600

# B
# Avem 18.288 esantioane masurate din ora in ora, deci 24 esantioane pe zi
# 18.288 / 24 = 762 zile

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def find_max_freq(signal, sample_rate):
    # FFT
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/sample_rate) # construieste vectorul de frecvente asociate coeficientilor FFT.

    # Luam doar frecventele pozitive
    positive_mask = freqs >= 0
    freqs_pos = freqs[positive_mask]
    spectrum_pos = np.abs(X[positive_mask]) # magnitudinea complexa
    
    # Sarim frecventa 0 (componenta DC)
    max_index = np.argmax(spectrum_pos[1:]) + 1  
    
    return freqs_pos[max_index]

train_df = pd.read_csv("Lab5/data/Train.csv", parse_dates=["Datetime"], dayfirst=True)
train_samples = train_df["ID"].values
train_signal = train_df["Count"].values
train_sample_rate = 1 / 3600

# C
print(find_max_freq(train_signal, train_sample_rate))

# D

def fft_magnitude(signal, sample_rate = 1):
    fourier = np.fft.fft(signal)
    magnitude = np.abs(fourier)[: len(fourier) // 2]

    frequencies = np.fft.fftfreq(len(signal), 1 / sample_rate)[: len(fourier) // 2]
    return fourier, frequencies, magnitude

fourier, frequencies, magnitude = fft_magnitude(train_signal, train_sample_rate)

fig, ax = plt.subplots(1)
fig.suptitle("Magnitude")

ax.plot(frequencies, magnitude, linewidth=0.5)

plt.show()

# E

def remove_continuous_component(signal):
    fourier, _, _ = fft_magnitude(signal)
    continous_component_value = np.abs(fourier[0]) / len(fourier)

    if continous_component_value == 0:
        raise ValueError("Nu exista componenta continua")

    return np.subtract(signal, continous_component_value)

train_signal_norm = remove_continuous_component(train_signal)

fig, ax = plt.subplots(1, figsize=(10, 5))
fig.suptitle("Fara componenta continua")

plt.xlim([np.min(train_samples), np.max(train_samples)])

ax.plot(train_samples, train_signal_norm, linewidth=0.75)

plt.show()

# F

def top_freq(signal, sample_rate):
    fourier, frequencies, magnitude = fft_magnitude(signal, sample_rate)

    sorted_magnitude_indices = np.argsort(magnitude)[-4:][::-1]
    return frequencies[sorted_magnitude_indices]

train_signal_norm = remove_continuous_component(train_signal)

top_4_freq = top_freq(train_signal_norm, train_sample_rate)

frequency_in_year = np.multiply(top_4_freq, 3600 * 24 * 365)

print("Top 4 frecvente: " + ', '.join(f'{freq:.12f}' for freq in top_4_freq))
print("Freq per year: " + ', '.join(f'{freq:.2f}' for freq in frequency_in_year))

# G
# number of days in a month
MONTH_SPAN = 28
# number of samples that represent one month of data
MONTH_SAMPLES_COUNT = 24 * MONTH_SPAN

sliced_df = train_df.iloc[1001:]
first_monday_index = sliced_df[sliced_df["Datetime"].dt.weekday == 0].index[0]

month_signal = train_df.loc[first_monday_index : first_monday_index + MONTH_SAMPLES_COUNT - 1, "Count"].values
month_samples = np.array([i + 1 for i in range(MONTH_SAMPLES_COUNT)])

fig, ax = plt.subplots()
fig.suptitle("Month Plot")

ax.plot(month_samples, month_signal)

ax.set_xlim([month_samples.min(), month_samples.max()])

plt.tight_layout()
plt.show()

# H

# I
# Pastram doar 5% cele mai mari amplitudini
filter_amount = 0.95

fourier = np.fft.fft(month_signal)
magnitude = np.abs(fourier)

threshold = np.percentile(magnitude, filter_amount * 100)
indices_to_remove = np.where(magnitude <= threshold)[0]

# Setare cu 0 indecsii sub threshold
for index in indices_to_remove:
    fourier[index] = 0 + 0j

filtered_signal = np.real(np.fft.ifft(fourier))

fig, ax = plt.subplots(2)
fig.suptitle("Filtred signal")

ax[0].plot(month_samples, filtered_signal)
ax[0].set_xlim([month_samples.min(), month_samples.max()])

ax[1].plot(month_samples, month_signal)
ax[1].set_xlim([month_samples.min(), month_samples.max()])

plt.tight_layout()
plt.show()