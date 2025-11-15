# Datorita legii numerelor mari, figura devine ca o distributie normala

import numpy as np
import matplotlib.pyplot as plt

B = 1.0
fs_list = [1.0, 1.5, 2.0, 4.0]
t_min, t_max = -3.0, 3.0
t_cont = np.linspace(t_min, t_max, 2000)

def semnal(t, B=1.0):
    return np.sinc(B * t) ** 2

def reconstruct_sinc(t_eval, t_samples, x_samples, Ts):
    """Reconstructia sinc"""
    
    arg = (t_eval[:, None] - t_samples[None, :]) / Ts
    sinc_vals = np.sinc(arg)
    return np.dot(sinc_vals, x_samples)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, fs in enumerate(fs_list):
    Ts = 1.0 / fs
    n_min = int(np.floor(t_min / Ts))
    n_max = int(np.ceil(t_max / Ts))
    n = np.arange(n_min, n_max + 1)
    t_samples = n * Ts
    x_samples = semnal(t_samples, B=B)
    x_hat = reconstruct_sinc(t_cont, t_samples, x_samples, Ts)

    ax = axes[i]
    ax.plot(t_cont, semnal(t_cont, B=B), label='x(t) continuu', color='black')
    ax.plot(t_samples, x_samples, 'o', label='esantioane')
    ax.plot(t_cont, x_hat, '--', label='x(t) reconstruit')
    ax.set_title(f'fs = {fs} Hz  (Ts = {Ts:.3f}s)')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Amplitudine')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
