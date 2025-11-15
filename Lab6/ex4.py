import numpy as np

N = 20
n = np.arange(N)
x = np.sin(2 * np.pi * n / N * 3)

# deplasare circulara
d = 5
y = np.roll(x, d)

X = np.fft.fft(x)
Y = np.fft.fft(y)

r = np.fft.ifft(X * Y)
q = np.fft.ifft(Y / X)

print(r)
print(q)

# Pentru primul caz, avem un semnal cu un maxim la pozitia d, iar in al doilea caz avem un impuls la pozitia d
