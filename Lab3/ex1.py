import numpy as np
from matplotlib import pyplot as plt

N = 8

fourier_matrix = np.zeros((N, N), dtype=np.complex64)

for i in range(N):
    for j in range(N):
        fourier_matrix[i][j] = np.exp(-2 * np.pi * 1j * j * i / N)


fourier_H_matrix = np.transpose(np.conjugate(fourier_matrix))

fhfm = np.matmul(fourier_H_matrix, fourier_matrix)
fhfm = np.subtract(fhfm, np.diag(np.full(N, fhfm[0, 0])))

# Daca norma Frobenius e aprox. 0, matricea este foarte aproape de matricea nula
# Toleranta pt erori numerice
eps = 1e-5
frobenius_norm = np.linalg.norm(fhfm, ord="fro")

print(f"Norm: {frobenius_norm}")
print(f"Is unitary: {0 - eps <= frobenius_norm <= 0 + eps}")

fig, axs = plt.subplots(N, sharex=True, sharey=True)
fig.suptitle("Componente Fourier")

for i in range(N):
    axs[i].plot(fourier_matrix[i].real)
    axs[i].plot(fourier_matrix[i].imag, linestyle="dashed")

plt.show()