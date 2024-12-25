import numpy as np

def gauss_method(A, B):
    augmented_matrix = np.column_stack((A, B))  
    rows, cols = augmented_matrix.shape

    
    for i in range(rows):
        max_row = np.argmax(np.abs(augmented_matrix[i:, i])) + i  
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]  
        augmented_matrix[i] /= augmented_matrix[i, i]  

        for j in range(i + 1, rows):
            augmented_matrix[j] -= augmented_matrix[j, i] * augmented_matrix[i]  

    x = np.zeros(rows)
    for i in range(rows - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] - np.sum(augmented_matrix[i, i + 1:cols - 1] * x[i + 1:])
    
    return x
