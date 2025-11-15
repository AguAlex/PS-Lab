import random
import numpy as np

seed = 0
np.random.seed(seed)
random.seed(seed)

N = 100
UPPER_LIMIT = 1000

p = np.random.randint(UPPER_LIMIT, size=N)
q = np.random.randint(UPPER_LIMIT, size=N)

# Convolutie directa in domeniul timpului
r_convolution = np.convolve(p, q)

# Conv cu fft in domeniul frecventei
p_fft = np.fft.fft(p, 2 * N - 1)
q_fft = np.fft.fft(q, 2 * N - 1)
F = np.multiply(p_fft, q_fft)

# np.real() se foloseste pentru a elimina componentele imaginare mici aparute din erori numerice ale FFT
r_fft = np.real(np.fft.ifft(F))

close_results = np.allclose(r_convolution, r_fft)
print(close_results)
# print(r_convolution)
# print(r_fft)