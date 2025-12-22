import numpy as np
from scipy.linalg import hankel

# EX1
def generate_time_series(N=1000):

    np.random.seed(42)
    t = np.linspace(0, 1, N)
    
    trend = 10 * t**2
    
    seasonality = 2 * np.sin(2 * np.pi * 10 * t) + 1.5 * np.cos(2 * np.pi * 25 * t)
    
    noise = np.random.normal(0, 0.5, N)
    
    y = trend + seasonality + noise
    
    return y, t

# EX2
def create_hankel(data, L):
    N = len(data)
    K = N - L + 1

    first_col = data[:L]
    last_row = data[L-1:]
    X = hankel(first_col, last_row)
    return X

y, t = generate_time_series()

L = 100 
X = create_hankel(y, L)

print(f"Dimensiunea seriei: {len(y)}")
print(f"Dimensiunea matricei Hankel X: {X.shape}")

# EX3
# SVD pentru X
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Valori proprii pentru XX^T
XXt = X @ X.T
eig_val_XXt, eig_vec_XXt = np.linalg.eigh(XXt)

# Sortam valorile proprii descrescator pentru a compara
idx = eig_val_XXt.argsort()[::-1]
eig_val_XXt = eig_val_XXt[idx]
eig_vec_XXt = eig_vec_XXt[:, idx]

# Verificarea relatiilor
print("Primele 5 valori singulare ale lui X:", S[:5])
print("Radacina patrata a primelor 5 valori proprii ale XX^T:", np.sqrt(np.abs(eig_val_XXt[:5])))

is_close = np.allclose(S[:5], np.sqrt(np.abs(eig_val_XXt[:5])), atol=1e-5)
print("Verificare relatie:", is_close)