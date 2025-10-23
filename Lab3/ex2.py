import matplotlib.pyplot as plt
import numpy as np

def normalized_distances(x):
    if len(x) == 0:
        return np.array([])

    if np.iscomplexobj(x):
        distances = np.abs(x)
    else:
        distances = [np.linalg.norm(point, ord=2) for point in x]

    max_distance = np.max(distances)
    return distances / max_distance if max_distance != 0 else distances

def winding_frequency_on_unit_circle(x, omegas):
    z = []
    for omega in omegas:
        zw = [x[i] * np.exp(-2 * np.pi * 1j * omega * i / len(x)) for i in range(len(x))]
        z.append(zw)
    return np.array(z)

n = 1000
t = 1000
nts = np.linspace(0, t, n)
f = 2
phi = np.pi / 2
signal = np.sin(3 * np.pi * f * nts + phi)

omegas = [1, 3, 4, 5, 10]
z = winding_frequency_on_unit_circle(signal, omegas)

fig, axs = plt.subplots(3, 2, figsize=(8, 12))
fig.suptitle("Winding frequency")

# Stilizare axele complexe
def setup_complex_axis(ax):
    ax.axhline(0, color="black")
    ax.axvline(0, color="black")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")

# Semnalul in timp
cmap = plt.get_cmap("inferno")
points = np.array(list(zip(nts, signal)))
colors = cmap(normalized_distances(points))
axs[0, 0].scatter(nts, signal, color=colors)
axs[0, 0].set_xlabel("Time")
axs[0, 0].set_ylabel("Amplitudine")
axs[0, 0].axhline(0, color="black")

# Cercul unitate pentru fiecare w
colors = cmap(normalized_distances(z[0]))
axs[0, 1].scatter(z[0].real, z[0].imag, color=colors)
setup_complex_axis(axs[0, 1])

for index in range(1, len(z)):
    i = (index - 1) // 2 + 1
    j = (index - 1) % 2
    colors = cmap(normalized_distances(z[index]))
    axs[i, j].scatter(z[index].real, z[index].imag, color=colors)
    setup_complex_axis(axs[i, j])

fig.tight_layout()
plt.show()
