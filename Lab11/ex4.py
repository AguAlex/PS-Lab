import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import hankel

def generate_time_series(N=1000):
    np.random.seed(42)
    t = np.linspace(0, 1, N)
    trend = 10 * t**2
    seasonality = 2 * np.sin(2 * np.pi * 10 * t) + 1.5 * np.cos(2 * np.pi * 25 * t)
    noise = np.random.normal(0, 0.5, N)
    y = trend + seasonality + noise
    return y, t

def create_hankel(data, L):
    N = len(data)
        
    first_col = data[:L]
    last_row = data[L-1:]
    X = hankel(first_col, last_row)
    return X

def diagonal_averaging(Matrix):
    """
    Realizeaza medierea pe diagonala.
    """
    L, K = Matrix.shape
    N = L + K - 1
    new_series = np.zeros(N)
    
    for k in range(N):
        diag_vals = []

        start_i = max(0, k - K + 1)
        end_i = min(L, k + 1)
        
        for i in range(start_i, end_i):
            j = k - i
            diag_vals.append(Matrix[i, j])
            
        if diag_vals:
            new_series[k] = np.mean(diag_vals)
            
    return new_series

def ssa_reconstruct(y, L, components):
    """
    1. Embedding (Hankel)
    2. SVD
    3. Grouping
    4. Diagonal Averaging
    """

    X = create_hankel(y, L)
    
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    X_rec = np.zeros_like(X)
    
    for i in components:

        component_matrix = S[i] * np.outer(U[:, i], Vt[i, :])
        X_rec += component_matrix
        
    y_rec = diagonal_averaging(X_rec)
    
    return y_rec


y, t = generate_time_series(N=200)
L = 100

trend_reconstruit = ssa_reconstruct(y, L, components=[0])

sezonalitate_reconstruita = ssa_reconstruct(y, L, components=[1, 2, 3, 4])

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, y, label='Date Originale (cu zgomot)', color='lightgray', linewidth=2)
plt.plot(t, trend_reconstruit, label='SSA Componenta 0 (Trend)', color='red', linewidth=2)
plt.title('SSA: Extragere Trend')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(t, sezonalitate_reconstruita, label='SSA Componentele 1-4 (Sezonalitate)', color='blue')
plt.title('SSA: Extragere Sezonalitate')
plt.legend()
plt.grid(True, alpha=0.3)

reziduuri = y - trend_reconstruit - sezonalitate_reconstruita
plt.subplot(3, 1, 3)
plt.plot(t, reziduuri, label='Reziduuri', color='green', alpha=0.7)
plt.title('SSA: Reziduuri')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()