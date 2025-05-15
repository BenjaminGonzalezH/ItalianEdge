######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.
from sklearn.metrics import adjusted_rand_score     # Pre-build ARI function.
from sklearn.metrics import rand_score              # Pre-build RI function.

######### Functions #########

"""
This block contains all main functions.
"""

def compute_rand_index(
        labels1: np.ndarray, 
        labels2: np.ndarray, 
        adjusted: bool = True
    ) -> float:
    """
    Compute Rand Index or Adjusted Rand Index between two clustering solutions.
    
    Parameters:
    - labels1 (np.ndarray): First clustering solution.
    - labels2 (np.ndarray): Second clustering solution.
    - adjusted (bool): If True, use Adjusted Rand Index. Otherwise, use Rand Index.
    
    Returns:
    - float: Similarity score.
    """
    if len(labels1) != len(labels2):
        raise ValueError("Both input arrays must have the same length.")
    
    return adjusted_rand_score(labels1, labels2) if adjusted else rand_score(labels1, labels2)

def process_RandValues(
        Matrix: list[np.ndarray], 
        n_threads: int, 
        adjusted: bool = True
    ) -> np.ndarray:
    """
    Compute the pairwise Rand or Adjusted Rand Index matrix for a set of clustering solutions.
    
    Parameters:
    - Matrix (list[np.ndarray] or 2D np.ndarray): List or 2D array of clustering solutions.
    - n_threads (int): Number of threads to use.
    - adjusted (bool): Whether to compute Adjusted Rand Index (True) or regular Rand Index (False).
    
    Returns:
    - np.ndarray: Similarity matrix (symmetric).
    """
    try:
        # Validar tipo y estructura
        if isinstance(Matrix, list):
            if not all(isinstance(row, np.ndarray) for row in Matrix):
                raise TypeError("If Matrix is a list, all elements must be NumPy arrays.")
        elif isinstance(Matrix, np.ndarray):
            if Matrix.ndim != 2:
                raise ValueError("Matrix must be a 2D NumPy array.")
            # Convertimos filas a lista de vectores
            Matrix = [Matrix[i, :] for i in range(Matrix.shape[0])]
        else:
            raise TypeError("Matrix must be a list of NumPy arrays or a 2D NumPy array.")

        if n_threads <= 0:
            raise ValueError("n_threads must be a positive integer.")

        n_solutions = len(Matrix)
        if n_solutions == 0:
            raise ValueError("Matrix must contain at least one solution.")

        # Inicializar matriz de similitud
        Similarity_Matrix = np.zeros((n_solutions, n_solutions))

        # Función auxiliar para el cálculo paralelo
        def compute_pairwise_rand(indices):
            i, j = indices
            score = compute_rand_index(Matrix[i], Matrix[j], adjusted=adjusted)
            return i, j, score

        # Índices del triángulo superior (incluyendo diagonal)
        indices = [(i, j) for i in range(n_solutions) for j in range(i, n_solutions)]

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            results = list(executor.map(compute_pairwise_rand, indices))

        # Rellenar matriz simétrica
        for i, j, value in results:
            Similarity_Matrix[i, j] = value
            Similarity_Matrix[j, i] = value

        return Similarity_Matrix

    except Exception as e:
        print(f"Error in process_RandValues: {e}")
        return np.array([])