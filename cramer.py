from determinants import det_4x4

def cramer_method(A, B):
    det_A = det_4x4(A)
    if det_A == 0:
        return "The system of equations is inconsistent."
    
    solutions = []
    for i in range(4):
        A_copy = A.copy()
        A_copy[:, i] = B
        det_A_i = det_4x4(A_copy)
        solutions.append(det_A_i / det_A)
    return solutions
