import numpy as np

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
def build_ar_matrix(series, p):
    """
    Construieste matricea X (regresori) si vectorul y (target)
    pentru un model AR de ordin p.
    """
    N = len(series)
    X = []
    y_target = []
    
    for i in range(p, N):
        # Luam ultimele p valori in ordine inversa: y[t-1], y[t-2], ...
        row = series[i-p:i][::-1] 
        X.append(row)
        y_target.append(series[i])
        
    return np.array(X), np.array(y_target)

def train_ar_ols(series, p):
    """
    Antreneaza modelul AR folosind Ordinary Least Squares (OLS).
    """
    X, y = build_ar_matrix(series, p)
    # Rezolvam w = (X^T X)^-1 X^T y
    w, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return w, X, y

if __name__ == "__main__":
    series, _ = generate_time_series()

    p = 30
    w_ols, X, y_true = train_ar_ols(series, p)

    print(f"S-au calculat {len(w_ols)} coeficienti.")
    print(f"Primii 5 coeficienti: {w_ols[:5]}")

    # Calculam predictia
    y_pred = X @ w_ols

    # Calculam eroarea MSE
    mse = np.mean((y_true - y_pred)**2)
    print(f"Eroarea (MSE): {mse}")