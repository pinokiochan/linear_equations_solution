import numpy as np

def det_3x3(matrix):
    return (matrix[0, 0] * (matrix[1,1] * matrix[2,2] - matrix[2,1] * matrix[1,2]) -
            matrix[0, 1] * (matrix[1,0] * matrix[2,2] - matrix[2,0] * matrix[1,2]) +
            matrix[0, 2] * (matrix[1,0] * matrix[2,1] - matrix[2,0] * matrix[1,1]))

def det_4x4(matrix):
    det = 0
    for col in range(4):
        submatrix = np.delete(np.delete(matrix, 0, axis=0), col, axis=1)
        sign = (-1) ** col
        det += sign * matrix[0,col] * det_3x3(submatrix)
    
    return det
