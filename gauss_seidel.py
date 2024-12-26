import numpy as np
from is_diagonally_dominant import is_diagonally_dominant

def gauss_seidel_method(A, b, max_iter=1000, tol=1e-10):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    iterations = 0  # Initialize iterations counter
    
    if not is_diagonally_dominant(A):
        P = np.eye(n)
        for i in range(n):
            max_row = i + np.argmax(np.abs(A[i:, i]))
            if max_row != i:
                A[[i, max_row]] = A[[max_row, i]]
                b[[i, max_row]] = b[[max_row, i]]
                P[[i, max_row]] = P[[max_row, i]]
    
    if np.any(np.abs(np.diag(A)) < 1e-10):
        return None, iterations, A  # Return None if diagonal elements are too small
    
    x = np.zeros(n)
    
    for iter_count in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        iterations += 1  # Increment iteration count
        
        if np.allclose(x, x_new, rtol=tol):
            return x_new, iterations, A  # Return the result, iteration count, and modified A
            
        if np.any(np.isnan(x_new)) or np.any(np.abs(x_new) > 1e6):
            return None, iterations, A  # Return None if invalid result encountered
            
        x = x_new
    
    return None, iterations, A  # Return None if max_iter is reached
