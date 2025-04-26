import numpy as np

def sorteig(A, order='descend'):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    # Determine sort order indices
    if order == 'descend':
        idx = np.argsort(eigenvalues)[::-1]
    else:
        idx = np.argsort(eigenvalues)
    
    # Create diagonal matrix of sorted eigenvalues
    D = np.diag(eigenvalues[idx])
    
    # Re-order the eigenvectors accordingly (columns are eigenvectors)
    V = eigenvectors[:, idx]
    
    return V, D

