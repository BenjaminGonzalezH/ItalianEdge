######### Libraries #########
import numpy as np                                      # Efficient Math Operations.
from concurrent.futures import ThreadPoolExecutor       # Thread Administration.
from sklearn.metrics import adjusted_rand_score         # Pre-build ARI function.
from sklearn.metrics import rand_score                  # Pre-build RI function.
from scipy.spatial.distance import pdist, squareform

######### Functions #########

"""
This block contains all main functions.
"""

def ComputeRandIndex(labels1: np.ndarray, labels2: np.ndarray, adjusted: bool = True) -> float:
    """
    Compute Rand Index or Adjusted Rand Index between two clustering solutions.
    """ 
    return adjusted_rand_score(labels1, labels2) if adjusted else rand_score(labels1, labels2)

def RandIndexSolutions(Matrix: np.ndarray, adjusted: bool = True) -> np.ndarray:
    """
    RandIndexSolutions(function): Compute the pairwise Rand Index or Adjusted Rand Index matrix using vectorization.
    """
    if not isinstance(Matrix, np.ndarray) or Matrix.ndim != 2:
        raise ValueError("Matrix must be a 2D NumPy array.")

    n_solutions = Matrix.shape[0]
    if n_solutions == 0:
        raise ValueError("Matrix must contain at least one solution.")

    # Vectorización con `pdist` y `squareform` para obtener la matriz simétrica
    Similarity_Matrix = squareform(pdist(Matrix, metric=lambda x, y: ComputeRandIndex(x, y, adjusted)))

    # Asignar valores de 1 en la diagonal (similitud consigo mismo)
    np.fill_diagonal(Similarity_Matrix, 1.0)

    return Similarity_Matrix