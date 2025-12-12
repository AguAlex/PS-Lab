import numpy as np
from ex1_2 import generate_time_series
from ex1_2 import train_ar_ols
from ex4 import find_roots_companion

def check_stationarity(roots):
    """
    Verifica daca toate radacinile sunt in cercul unitate.
    """
    magnitudes = np.abs(roots)
    max_mag = np.max(magnitudes)
    
    is_stationary = max_mag < 1.0
    return is_stationary, max_mag

if __name__ == "__main__":

    series, _ = generate_time_series()
 
    p = 10
    w_ols, _, _ = train_ar_ols(series, p)
    
    roots = find_roots_companion(w_ols)
    
    is_stat, max_val = check_stationarity(roots)
    
    print(f"Cea mai mare magnitudine a radacinii:", max_val)
    print(is_stat)