import numpy as np
from is_diagonally_dominant import is_diagonally_dominant

def jacobi_method(A, b, max_iter=1000, tol=1e-10):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    if not is_diagonally_dominant(A):

        P = np.eye(n)
        for i in range(n):
            max_row = i + np.argmax(np.abs(A[i:, i]))
            if max_row != i:
                A[[i, max_row]] = A[[max_row, i]]
                b[[i, max_row]] = b[[max_row, i]]
                P[[i, max_row]] = P[[max_row, i]]
    
    D = np.diag(A)
    if np.any(np.abs(D) < 1e-10):
        return None  
        
    R = A - np.diagflat(D)
    x = np.zeros(n)
    
    for i in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        if np.allclose(x, x_new, rtol=tol):
            return x_new
        x = x_new
        
        if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e6):
            return None
            
    return None  