import numpy as np
import scipy.io.wavfile as wav
import sounddevice

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

fs = 44100

# (a)
sin_1600 = a(t1)

# (b)
sin_800 = b(t2)

# (c)
signal = c(t3)

# (d)
square = d(t3)

wav.write(r"C:\Users\Agu\Desktop\sin_1600.wav", fs, sin_1600)
file_rate, audio_file = wav.read(r"C:\Users\Agu\Desktop\sin_1600.wav")
sounddevice.play(audio_file, samplerate = file_rate)

wav.write(r"C:\Users\Agu\Desktop\signal.wav", fs, sin_800)
file_rate, audio_file = wav.read(r"C:\Users\Agu\Desktop\signal.wav")
sounddevice.play(audio_file, samplerate = file_rate)

wav.write(r"C:\Users\Agu\Desktop\sin_800.wav", fs, signal)
file_rate, audio_file = wav.read(r"C:\Users\Agu\Desktop\sin_800.wav")
sounddevice.play(audio_file, samplerate = file_rate)

wav.write(r"C:\Users\Agu\Desktop\square.wav", fs, square)
file_rate, audio_file = wav.read(r"C:\Users\Agu\Desktop\square.wav")
sounddevice.play(audio_file, samplerate = file_rate)
