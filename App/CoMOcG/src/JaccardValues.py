######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import rand_score
import itertools

######### Functions #########

"""
This block contains all main functions.
"""

def compute_jaccard(A, B):
    A = np.array(A)
    B = np.array(B)
    n = len(A)

    # Creamos matrices booleanas de comparación
    same_A = (A[:, None] == A[None, :])
    same_B = (B[:, None] == B[None, :])

    # Tomamos solo la parte superior triangular para evitar duplicados (i<j)
    upper_triangle = np.triu_indices(n, k=1)

    same_A_upper = same_A[upper_triangle]
    same_B_upper = same_B[upper_triangle]

    # Condiciones para r, u, v
    r = np.sum(same_A_upper & same_B_upper)
    u = np.sum(same_A_upper & ~same_B_upper)
    v = np.sum(~same_A_upper & same_B_upper)

    if (r + u + v) == 0:
        return 0

    return r / (r + u + v)

def process_JaccardValues(
        Matrix: list[np.ndarray], 
        n_threads: int) -> np.ndarray:
    """
    process_JaccardValues(function): Compute the pairwise Jaccard similarity matrix for a set of solutions.

    Parameters:
    - Matrix (list[np.ndarray]] or np.ndarray): Matrix of cluster solutions.
    - n_threads (int): Number of threads to use for parallel computation.

    Returns:
    - Jaccard_Matrix (np.ndarray): Pairwise Jaccard similarity matrix for the solutions.
    """
    # Check input.
    if isinstance(Matrix, list):
        if not all(isinstance(row, list) for row in Matrix):
            raise ValueError("Matrix has to be a list of ndarrays.")
    elif isinstance(Matrix, np.ndarray):
        if Matrix.ndim != 2:
            raise ValueError("Matrix is Numpy array 2D")
    else:
        raise ValueError("Matrix has to be a list of ndarrays or NumPy array 2D.")
    
    if n_threads <= 0:
        raise ValueError("n_threads must be a positive integer.")
    
    # Create a Jaccard Matrix.
    n_solutions = len(Matrix)
    Jaccard_Matrix = np.zeros((n_solutions, n_solutions))

    # Aux function to handle solution's index.
    def compute_pairwise_proportion(indices):
        i, j = indices
        return i, j, compute_jaccard(Matrix[i], Matrix[j])
    
    # Create Combination of index.
    indices = [(i, j) for i in range(n_solutions) for j in range(i, n_solutions)]
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(compute_pairwise_proportion, indices))
    
    # Fill symetric jaccard values matrix.
    for i, j, proportion in results:
        Jaccard_Matrix[i, j] = proportion
        Jaccard_Matrix[j, i] = proportion
    
    return Jaccard_Matrix

def Jaccar_similarityClusters(
        Solution1: list[set], 
        Solution2: list[set]) -> np.ndarray:
    """
    Jaccar_similarityClusters(function): Compute the Jaccard similarity matrix for
    two specifict solutions, this works to recognize similar clusters between two
    solutiions.

    Parameters:
    - Solution1 (list[set]): Clusters of the first solution as sets.
    - Solution2 (list[set]): Clusters of the second solution as sets.

    Returns:
    - MatrixJaccard (np.ndarray): Jaccard similarity matrix for the 
    solution's clusters.
    """
    # Construction of jaccard matrix.
    n = len(Solution1)
    MatrixJaccard = np.zeros((n, n))

    # Compare clusters of two solutions.
    for i in range(len(Solution1)):
        for j in range(i, len(Solution2)):
            union = len(Solution1[i] | Solution2[j])
            intersection = len(Solution1[i] & Solution2[j])
            if union == 0:
                MatrixJaccard[i, j] = 0
                MatrixJaccard[j, i] = 0
            else:
                Jaccard = intersection / union
                MatrixJaccard[i, j] = Jaccard
                MatrixJaccard[j, i] = Jaccard

    return MatrixJaccard

def compute_ari(Solution1: np.ndarray, Solution2: np.ndarray) -> float:
    """
    compute_ari(function): Compute the Adjusted Rand Index (ARI) for two solutions.

    Parameters:
    - Solution1 (ndarray): First solution vector (cluster assignments).
    - Solution2 (ndarray): Second solution vector (cluster assignments).

    Returns:
    - ari_index (float): Adjusted Rand Index between the two vectors.
    """
    # Check length.
    if len(Solution1) != len(Solution2):
        raise ValueError("Both vectors must have the same length.")
    
    return adjusted_rand_score(Solution1, Solution2)

def process_ARIValues(Matrix: list[np.ndarray], n_threads: int) -> np.ndarray:
    """
    process_ARIValues(function): Compute the pairwise Adjusted Rand Index matrix for a set of solutions.

    Parameters:
    - Matrix (list[np.ndarray]] or np.ndarray): Matrix of cluster solutions.
    - n_threads (int): Number of threads to use for parallel computation.

    Returns:
    - ARI_Matrix (np.ndarray): Pairwise Adjusted Rand Index matrix for the solutions.
    """
    # Check input.
    if isinstance(Matrix, list):
        if not all(isinstance(row, np.ndarray) for row in Matrix):
            raise ValueError("Matrix has to be a list of ndarrays.")
    elif isinstance(Matrix, np.ndarray):
        if Matrix.ndim != 2:
            raise ValueError("Matrix must be a 2D NumPy array")
    else:
        raise ValueError("Matrix has to be a list of ndarrays or a 2D NumPy array.")
    
    if n_threads <= 0:
        raise ValueError("n_threads must be a positive integer.")
    
    # Create an Adjusted Rand Index Matrix.
    n_solutions = len(Matrix)
    ARI_Matrix = np.zeros((n_solutions, n_solutions))

    # Aux function to handle solution's index.
    def compute_pairwise_ari(indices):
        i, j = indices
        return i, j, compute_ari(Matrix[i], Matrix[j])
    
    # Create Combination of indices.
    indices = [(i, j) for i in range(n_solutions) for j in range(i, n_solutions)]
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(compute_pairwise_ari, indices))
    
    # Fill symmetric ARI values matrix.
    for i, j, ari_value in results:
        ARI_Matrix[i, j] = ari_value
        ARI_Matrix[j, i] = ari_value
    
    return ARI_Matrix

def compute_ri(Solution1: np.ndarray, Solution2: np.ndarray) -> float:
    """
    compute_ri(function): Compute the Rand Index (RI) for two solutions.

    Parameters:
    - Solution1 (ndarray): First solution vector (cluster assignments).
    - Solution2 (ndarray): Second solution vector (cluster assignments).

    Returns:
    - ri_index (float): Rand Index between the two vectors.
    """
    # Check length.
    if len(Solution1) != len(Solution2):
        raise ValueError("Both vectors must have the same length.")
    
    return rand_score(Solution1, Solution2)

def process_RIValues(Matrix: list[np.ndarray], n_threads: int) -> np.ndarray:
    """
    process_RIValues(function): Compute the pairwise Rand Index matrix for a set of solutions.

    Parameters:
    - Matrix (list[np.ndarray]] or np.ndarray): Matrix of cluster solutions.
    - n_threads (int): Number of threads to use for parallel computation.

    Returns:
    - RI_Matrix (np.ndarray): Pairwise Rand Index matrix for the solutions.
    """
    # Check input.
    if isinstance(Matrix, list):
        if not all(isinstance(row, np.ndarray) for row in Matrix):
            raise ValueError("Matrix has to be a list of ndarrays.")
    elif isinstance(Matrix, np.ndarray):
        if Matrix.ndim != 2:
            raise ValueError("Matrix must be a 2D NumPy array")
    else:
        raise ValueError("Matrix has to be a list of ndarrays or a 2D NumPy array.")
    
    if n_threads <= 0:
        raise ValueError("n_threads must be a positive integer.")
    
    # Create Rand Index Matrix.
    n_solutions = len(Matrix)
    RI_Matrix = np.zeros((n_solutions, n_solutions))

    # Aux function to handle solution's index.
    def compute_pairwise_ri(indices):
        i, j = indices
        return i, j, compute_ri(Matrix[i], Matrix[j])
    
    # Create Combination of indices.
    indices = [(i, j) for i in range(n_solutions) for j in range(i, n_solutions)]
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(compute_pairwise_ri, indices))
    
    # Fill symmetric RI matrix.
    for i, j, ri_value in results:
        RI_Matrix[i, j] = ri_value
        RI_Matrix[j, i] = ri_value
    
    return RI_Matrix