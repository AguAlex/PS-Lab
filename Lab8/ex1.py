import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

np.random.seed(0)

def sin_signal(A, f, t, phi, n):
    nts = np.linspace(0, t, n)
    return A * np.sin(2 * np.pi * f * nts + phi)

# A

N = 1000
ts = np.linspace(0, 1, N)
trend_function = lambda x: x**2
trend = np.array([trend_function(x) for x in ts])
season = sin_signal(0.1, 5, 1, 0, N) + sin_signal(0.2, 3, 1, 0, N)
noise = np.random.normal(0, 0.01, N)
series = trend + season + noise

fig, axs = plt.subplots(4)
fig.suptitle("A) Time Series")
index_title = 0
titles=["Original", "Trend", "Season", "Noise"]

for ax, y in zip(axs.flat, [series, trend, season, noise]):
    ax.plot(ts, y)

    ax.set_title(titles[index_title])

    index_title += 1

plt.tight_layout()
plt.show()

# B
autocorr = np.correlate(series - np.mean(series), series - np.mean(series), mode='full')[:N]

# Normalizam si luam doar partea pozitiva
autocorr = autocorr[autocorr.size // 2:]
autocorr = autocorr / autocorr[0]

plt.figure(figsize=(10, 4))
plt.plot(np.arange(-len(autocorr), 0), autocorr)
plt.title("Functia de Autocorelatie")
plt.xlabel("Lag")
plt.ylabel("Corelatie")
plt.grid(True)
plt.show()

# C

def fit_ar_model(series, p):
    # Trebuie sa rezolvam X * w = y_target
    # X este matricea de valori trecute (lag-uri)
    # y_target este valoarea curenta
    
    X = []
    y_target = []
    
    for i in range(p, len(series)):
        X.append(series[i-p:i][::-1]) # [y[t-1], y[t-2], ..., y[t-p]]
        y_target.append(series[i])
        
    X = np.array(X)
    y_target = np.array(y_target)
    
    # Rezolvam cu Least Squares (Regresie Liniara); w = (X^T X)^-1 X^T y
    w, residuals, rank, s = np.linalg.lstsq(X, y_target, rcond=None)
    
    return w, X, y_target

p_demo = 20
weights, X_demo, y_true_demo = fit_ar_model(y, p_demo)

y_pred_demo = np.dot(X_demo, weights)

# Seria originala si predictiile
plt.figure(figsize=(12, 5))

zoom_start = 800
plt.plot(np.arange(zoom_start, N), y[zoom_start:], label='Original', alpha=0.7)
plt.plot(np.arange(zoom_start, N), y_pred_demo[zoom_start-p_demo:], label=f'Predictie AR(p={p_demo})', linestyle='--')
plt.title(f"Model AR({p_demo}): Original vs Predictie")
plt.legend()
plt.show()

# D
# Vom imparti datele in Train si Test pentru a valida corect

train_size = int(N * 0.8)
train_data = y[:train_size]
test_data = y[train_size:]

best_p = 0
best_mse = float('inf')
max_p_search = 50

mse_history = []

for p_val in range(1, max_p_search + 1):
    # Antrenare
    w, _, _ = fit_ar_model(train_data, p_val)
    
    # Testare
    # Pentru a prezice y[t], folosim valorile reale anterioare (y[t-1]...y[t-p])
    
    errors = []
    # Parcurgem setul de test
    for i in range(len(test_data)):
        
        full_history = np.concatenate([train_data, test_data[:i]])
        last_p_values = full_history[-p_val:][::-1]
        
        pred = np.dot(last_p_values, w)
        actual = test_data[i]
        errors.append((pred - actual)**2)
    
    mse = np.mean(errors)
    mse_history.append(mse)
    
    if mse < best_mse:
        best_mse = mse
        best_p = p_val

print(f"Cel mai bun ordin AR gasit este p = {best_p}")
print(f"Eroarea MSE minima: {best_mse:.4f}")

plt.figure(figsize=(10, 4))
plt.plot(range(1, max_p_search + 1), mse_history, marker='o')
plt.title("Performanta modelului (MSE) in functie de p")
plt.xlabel("Ordinul p")
plt.ylabel("MSE (Eroare)")
plt.axvline(best_p, color='r', linestyle='--', label=f'Best p={best_p}')
plt.legend()
plt.grid()
plt.show()