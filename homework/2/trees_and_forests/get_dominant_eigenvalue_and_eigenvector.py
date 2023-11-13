import numpy as np
def  get_dominant_eigenvalue_and_eigenvector(matrix, num_iterations):
    n = matrix.shape[0]
    x = np.random.rand(n)
    
    for i in range(num_iterations):
        y = np.dot(matrix, x)
        
        x = y / np.linalg.norm(y)
        eigenvalue = np.dot(x, np.dot(matrix, x))
        
    return float(eigenvalue), x