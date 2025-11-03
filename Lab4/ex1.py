# {'my_dft': [0.02262450000125682, 0.09189890000016021, 0.3627061000006506, 1.5978994000015518, 7.289478000000599, 30.893117300000085, 119.29981129999942], 
#  'np_fft': [0.0029220000014902325, 6.730000131938141e-05, 5.649999911838677e-05, 0.00011960000119870529, 0.00020009999934700318, 0.00019319999955769163, 0.000316400000883732]}

import time
import matplotlib.pyplot as plt
import numpy as np

def fourier_transform(x):

    n = len(x)
    X = np.zeros(n, dtype=np.complex128)

    for m in range(n):
        component_sum = 0

        for k in range(n):
            component_sum += x[k] * np.exp(-2 * np.pi * 1j * k * m / n)

        X[m] = component_sum

    return X

def calcutate_times(N):
    times = {"my_dft": [], "np_fft": []}

    for n in N:
        signal = np.random.uniform(-1, 1, n)

        start_my_dft = time.perf_counter()
        fourier_transform(signal)
        end_my_dft = time.perf_counter()

        start_np_fft = time.perf_counter()
        np.fft.fft(signal)
        end_np_fft = time.perf_counter()

        times["my_dft"].append(end_my_dft - start_my_dft)
        times["np_fft"].append(end_np_fft - start_np_fft)

    return times

N = [128, 256, 512, 1024, 2048, 4096, 8192]
times = calcutate_times(N)
print(times)

fig, axs = plt.subplots(1)
fig.suptitle("Time")
axs.plot(N, times["my_dft"], color="r")
axs.plot(N, times["np_fft"], color="b")

plt.show()
