######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.

######### Functions #########

def compute_jaccard(Solution1: np.ndarray, Solution2: np.ndarray) -> float:
    """
    compute_jaccard(function): Compute the Jaccard similarity index for two solutions.

    Parameters:
    - Solution1 (ndarray): First solution vector.
    - Solution2 (ndarray): Second solution vector.

    Returns:
    - jaccard_index (float): Jaccard similarity index between the two vectors.
    """
    # Check length.
    if len(Solution1) != len(Solution2):
        raise ValueError("Ambos vectores deben tener la misma longitud.")
    
    # Jaccard Value calculus.
    intersection = np.sum(np.minimum(Solution1, Solution2))
    union = np.sum(np.maximum(Solution1, Solution2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def process_JaccardValues(Matrix: list[np.ndarray], n_threads: int):
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

def Jaccar_similarityClusters(Solution1: list[set], Solution2: list[set]):
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
