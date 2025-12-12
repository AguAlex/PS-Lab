import numpy as np

def find_roots_companion(coeffs):

    p = len(coeffs)
    if p == 0:
        return np.array([])
        
    # Construim Matricea Companion
    # Prima linie contine coeficientii
    C = np.zeros((p, p))
    C[0, :] = coeffs
    
    # Subdiagonala cu 1
    if p > 1:
        np.fill_diagonal(C[1:, :], 1)
        
    # Calculam valorile proprii
    roots = np.linalg.eigvals(C)
    
    return roots

# Test cu un polinom cunoscut: z^2 - 3z + 2 = 0 -> Radacini: 1 si 2
# w = [3, -2] (deoarece z^2 = 3z - 2)
test_coeffs = np.array([3.0, -2.0]) 
roots = find_roots_companion(test_coeffs)

print("Roots:", roots)