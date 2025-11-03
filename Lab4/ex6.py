import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile

# Segmentare semnal in ferestre suprapuse
def segment_signal(signal):

    segmented_signal = []
    
    window_size = len(signal) // 100
    step = window_size // 2 # suprapunere de 50%

    for i in range(0, len(signal) - window_size + 1, step):
        segmented_signal.append(signal[i : i + window_size])

    return np.array(segmented_signal)

sample_rate, audio_signal = wavfile.read("C:/Users/Agu/Desktop/vocals.wav")

# Conversie la mono pt erori
if audio_signal.ndim > 1:
    audio_signal_mono = audio_signal.mean(axis=1)
else:
    audio_signal_mono = audio_signal

# Conversie la float pentru precizie pentru fft
audio_signal_mono = audio_signal_mono.astype(np.float64)

segmented_signal = segment_signal(audio_signal_mono)

fft_signal = np.fft.fft(segmented_signal, axis=1)

# Luam prima jumatate (Nyquist) si modulul
spectrogram_magnitude = np.abs(fft_signal)[:, : fft_signal.shape[1] // 2 + 1]

# Scalare logaritmica pt dB
spectrogram_db = 10 * np.log10(spectrogram_magnitude + 1e-10)

# Transpunere pt afisare
spectrogram_final = np.transpose(spectrogram_db)

total_time = len(audio_signal_mono) / sample_rate 

# Extragem frecvențele 
frequencies = np.fft.fftfreq(
    segmented_signal.shape[1], d=1 / sample_rate
)[: spectrogram_magnitude.shape[1]]

# Calcul limite de culori
v_max = np.max(spectrogram_final)
v_min = v_max - 60 # Pragul de zgomot

plt.figure(figsize=(12, 6))
plt.imshow(
    spectrogram_final,
    aspect="auto",
    extent=(0, total_time, frequencies[0], frequencies[-1]),
    origin="lower", 
    cmap="magma",
    vmin=v_min, 
    vmax=v_max, 
)
plt.colorbar(label="Magnitudine (dB)")
plt.xlabel("Timp")
plt.ylabel("Frecventa")

plt.title("Spectrograma")

plt.show()