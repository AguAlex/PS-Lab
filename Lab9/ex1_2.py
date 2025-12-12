import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

np.random.seed(0)

def sin_signal(A, f, t, phi, n):
    nts = np.linspace(0, t, n)
    return A * np.sin(2 * np.pi * f * nts + phi)

# 1

N = 1000
ts = np.linspace(0, 1, N)
trend_function = lambda x: x**2
trend = np.array([trend_function(x) for x in ts])
season = sin_signal(0.1, 5, 1, 0, N) + sin_signal(0.2, 3, 1, 0, N)
noise = np.random.normal(0, 0.01, N)
series = trend + season + noise

train_size = int(N * 0.85)
train_data = series[:train_size]
test_data = series[train_size:]
test_idx = np.arange(train_size, N)

# 2
# A) Mediere Simpla
best_alpha = 0
best_mse_ses = float('inf')
best_ses_model = None

# Cautam alpha intre 0.01 si 0.99
for alpha in np.linspace(0.01, 0.99, 50):

    model = SimpleExpSmoothing(train_data).fit(smoothing_level=alpha, optimized=False)
    pred = model.forecast(len(test_data))
    mse = mean_squared_error(test_data, pred)
    
    if mse < best_mse_ses:
        best_mse_ses = mse
        best_alpha = alpha
        best_ses_model = pred

print(f"Simple Exp Smoothing Best Alpha: {best_alpha}, MSE: {best_mse_ses}")

# B) Mediere dubla
model_double = ExponentialSmoothing(train_data, trend='add', seasonal=None).fit()
pred_double = model_double.forecast(len(test_data))
mse_double = mean_squared_error(test_data, pred_double)

# C) Mediere tripla
seasonal_period = int(N / 6) 
model_triple = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=seasonal_period).fit()
pred_triple = model_triple.forecast(len(test_data))
mse_triple = mean_squared_error(test_data, pred_triple)

print(f"Double Exp Smoothing MSE: {mse_double}")
print(f"Triple Exp Smoothing MSE: {mse_triple}")

plt.figure(figsize=(12, 5))
plt.plot(test_idx, test_data, label="Date Reale (Test)", color='black', alpha=0.5)
plt.plot(test_idx, best_ses_model, label=f"Simple (alpha={best_alpha:.2f})", linestyle='--')
plt.plot(test_idx, pred_double, label="Double", linestyle='--')
plt.plot(test_idx, pred_triple, label="Triple", linewidth=2)
plt.title("Exercitiul 2: Comparatie Medieri Exponentiale")
plt.legend()
plt.show()