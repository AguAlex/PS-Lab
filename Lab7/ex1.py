import matplotlib.pyplot as plt
import numpy as np

def compute_image(f):
    matrix = np.zeros((256, 256))

    for i in range(256):
        for j in range(256):
            matrix[i][j] = f(i, j)

    return matrix

def f1(x, y):
    return np.sin(2 * np.pi * x + 3 * np.pi * y)

def f2(x, y):
    return np.sin(4 * np.pi * x) + np.cos(6 * np.pi * y)

def f3(x, y):
    return 1 if (x == 0 and y == 5) or (x == 0 and y == 256 - 5) else 0

# f4 e f3 transpus
def f4(x, y):
    return f3(y, x)

def f5(x, y):
    return f3(x, y) + f4(x, y)

matrix1 = compute_image(f1)
plt.imshow(matrix1, cmap="grey")
plt.show()

matrix2 = compute_image(f2)
plt.imshow(matrix2, cmap="grey")
plt.show()

matrix3 = compute_image(f3)
reverse_matrix3 = np.real(np.fft.ifft2(matrix3))

plt.imshow(reverse_matrix3, cmap="grey")
plt.show()

matrix4 = compute_image(f4)
reverse_matrix4 = np.real(np.fft.ifft2(matrix4))
plt.imshow(reverse_matrix4, cmap="grey")
plt.show()

matrix5 = compute_image(f5)
reverse_matrix5 = np.real(np.fft.ifft2(matrix5))
plt.imshow(reverse_matrix5, cmap="grey")
plt.show()
