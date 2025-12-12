import numpy as np
import cvxopt
from ex1_2 import generate_time_series
from ex1_2 import build_ar_matrix

# 1. Metoda greedy
def train_ar_greedy(series, p, k_features):
    X, y = build_ar_matrix(series, p)
    n_samples, n_features = X.shape
    
    # Initializare
    w = np.zeros(n_features)
    residual = y.copy()
    indices = []
    
    for _ in range(k_features):
        # Calculam corelatia dintre reziduu si fiecare coloana nefolosita din X
        correlations = X.T @ residual
        
        # Gasim indicele cu cea mai mare corelatie absoluta
        best_idx = np.argmax(np.abs(correlations))
        
        if best_idx in indices:
            break
            
        indices.append(best_idx)
        
        # Rezolvam OLS doar pe indicii selectati
        X_subset = X[:, indices]
        w_subset, _, _, _ = np.linalg.lstsq(X_subset, y, rcond=None)
        
        # Updatam reziduul
        y_pred = X_subset @ w_subset
        residual = y - y_pred
        
        # Punem coeficientii inapoi in vectorul mare
        w[indices] = w_subset
        
    return w

# 2. Metoda L1 Regularization (Lasso) cu CVXOPT
def train_ar_l1_cvxopt(series, p, lam):
    X, y = build_ar_matrix(series, p)
    m, n = X.shape
    
    XTX = X.T @ X
    P_block = np.block([[XTX, -XTX], [-XTX, XTX]])
    P = cvxopt.matrix(2 * P_block)
    
    XTy = X.T @ y
    q_top = -2 * XTy + lam
    q_bot = 2 * XTy + lam
    q = cvxopt.matrix(np.concatenate([q_top, q_bot]))
    
    G = cvxopt.matrix(-np.eye(2 * n))
    h = cvxopt.matrix(np.zeros(2 * n))
    
    # Rezolvare
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h)
    
    # Recuperam solutia w = u - v
    uv = np.array(sol['x']).flatten()
    u = uv[:n]
    v = uv[n:]
    w = u - v
    
    # Setam valorile foarte mici la 0 (numerical noise)
    w[np.abs(w) < 1e-4] = 0
    
    return w

if __name__ == "__main__":
    series, _ = generate_time_series()
    p = 50

    # A) Greedy
    k = 5 # Vrem doar 5 coeficienti nenuli
    w_greedy = train_ar_greedy(series, p, k)
    print(f"Greedy - Coeficienti nenuli: {np.count_nonzero(w_greedy)}")

    # B) L1 (Lasso)
    lam = 10.0 # Factorul de regularizare
    w_l1 = train_ar_l1_cvxopt(series, p, lam)
    print(f"L1 - Coeficienti nenuli: {np.count_nonzero(w_l1)}")

    # Comparatie coeficienti
    print("\nVerificare Sparsitate (primii 10 coef):")
    print(f"Greedy: {np.round(w_greedy[:10], 2)}")
    print(f"L1:     {np.round(w_l1[:10], 2)}")