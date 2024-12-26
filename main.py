import numpy as np
from cramer import cramer_method
from gauss import gauss_method
from jacobi import jacobi_method
from gauss_seidel import gauss_seidel_method

def compare_methods(A, b):
    print("Original system:")
    print("Matrix A:")
    print(A)
    print("Vector b:", b)
    print("\nSolutions by different methods:")
    
    methods = [
        ("Cramer", cramer_method),
        ("Gauss", gauss_method),
        ("Jacobi", jacobi_method),
        ("Gauss-Seidel", gauss_seidel_method)
    ]
    
    for name, method in methods:
        try:
            if  name == "Gauss":
                result, modified_matrix = method(A, b)
                print(f"\nMethod {name}:")
                print(f"Solution: {result}")
                print(f"Modified Matrix A: \n{modified_matrix}")
            if  name == "Cramer":
                result = method(A, b)
                print(f"\nMethod {name}:")
                print(f"Solution: {result}")

            else:
                result, iterations, modified_matrix = method(A, b)
                print(f"\nMethod {name}:")
                print(f"Solution: {result}")
                print(f"Iterations: {iterations}")
                print(f"Modified Matrix A: \n{modified_matrix}")
        except Exception as e:
            print(f"\nMethod {name}: Error - {str(e)}")
    

if __name__ == "__main__":
    # Test system
    A = np.array([
        [3, -5, 47, 20],
        [11, 16, 17, 10],
        [56, 22, 11, -18],
        [17, 66, -12, 7]
    ], dtype=float)
    
    b = np.array([18, 26, 34, 82], dtype=float)
    
    compare_methods(A, b)
