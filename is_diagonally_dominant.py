import numpy as np
def is_diagonally_dominant(A):
    D = np.abs(np.diag(A))
    S = np.sum(np.abs(A), axis=1) - D
    return np.all(D > S)