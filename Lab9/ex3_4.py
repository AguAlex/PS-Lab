from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from ex1_2 import train_data, test_data, test_idx

best_aic = float('inf')
best_order = (0, 0, 0)
best_model_predictions = None

max_p_q = 4 

aic_history = []
pq_combinations = []

print(f"Cautare parametri optimi (Grid Search 0-{max_p_q})...")

for p in range(max_p_q + 1):
    for q in range(max_p_q + 1):
        if p == 0 and q == 0:
            continue
            
        try:
            # Definim modelul ARIMA(p, 0, q)
            model = ARIMA(train_data, order=(p, 0, q))
            model_fit = model.fit()
            
            current_aic = model_fit.aic
            
            aic_history.append(current_aic)
            pq_combinations.append(f"({p},{q})")
            
            if current_aic < best_aic:
                best_aic = current_aic
                best_order = (p, 0, q)

                # Facem predictii pe setul de test
                best_model_predictions = model_fit.forecast(steps=len(test_data))
                
        except:
            continue

print(f"Cel mai bun model gasit: ARMA{best_order} cu AIC: {best_aic:.2f}")

# Calculam MSE pentru cel mai bun model ARMA
mse_arma = mean_squared_error(test_data, best_model_predictions)
print(f"MSE pe setul de test pentru Best ARMA: {mse_arma:.4f}")
