import numpy as np
from cramer import cramer_method
from gauss import gauss_method
from jacobi import jacobi_method
from gauss_seidel import gauss_seidel_method
from determinants import det_4x4

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
    
    results = {}
    for name, method in methods:
        try:
            result = method(A, b)
            if result is None:
                print(f"\nMethod {name}: Did not converge")
            else:
                result = np.array(result)
                results[name] = result
                print(f"\nMethod {name}:")
                print(result)
        except Exception as e:
            print(f"\nMethod {name}: Error - {str(e)}")
    
    # Compare results
    print("\nComparison of results:")
    if len(results) > 1:
        print("Maximum difference between converged methods:")
        max_diff = max(np.max(np.abs(results[m1] - results[m2])) 
                       for m1 in results for m2 in results if m1 != m2)
        print(f"Maximum difference: {max_diff}")
        print(f"Converged methods: {', '.join(results.keys())}")
    else:
        print("Not enough converged methods to compare")

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