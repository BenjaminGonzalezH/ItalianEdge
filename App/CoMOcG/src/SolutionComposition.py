######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from concurrent.futures import ThreadPoolExecutor   # Threads Administration.

######### Functions #########

"""
This block contains all main functions.
"""

def compute_proportion_genesolution(
        Solution1: np.ndarray, 
        Solution2: np.ndarray) -> float:
    """
    compute_proportion_genesolution(function): 
    Compute the proportion of identical values between two gene solutions.

    Parameters:
    - Solution1 (np.ndarray): First gene solution array.
    - Solution2 (np.ndarray): Second gene solution array.

    Returns:
    - proportion (float): Proportion of identical values between the two solutions.
    """
    
    # Checking input.
    if len(Solution1) == 0 or len(Solution2) == 0:
        raise ValueError("Solutions cannot be empty.")
    if len(Solution1) != len(Solution2):
        raise ValueError("Solutions must have the same length.")
    
    # Calculate the proportion.
    count = np.sum(Solution1 == Solution2)  # Count matches.
    proportion = count / len(Solution1)

    return proportion

def process_proportion_genessolution(
        Matrix: list[np.ndarray], 
        n_threads: int) -> np.ndarray:
    """
    process_proportion_genessolution(function): 
    Compute the Composition matrix between all solutions in concurrency.

    Parameters:
    - Matrix (list or np.ndarray): Matrix of solutions (list of lists or NumPy array).
    - n_threads (int): Number of threads to 'parallelize' the computation.

    Returns:
    - CompositionMatrix (np.ndarray): Composition matrix with the proportion of matches.
    """
    # Input validation.
    if isinstance(Matrix, list):
        if not all(isinstance(row, list) for row in Matrix):
            raise ValueError("Matrix must be a list of lists.")
    elif isinstance(Matrix, np.ndarray):
        if Matrix.ndim != 2:
            raise ValueError("Matrix must be a 2D NumPy array.")
    else:
        raise ValueError("Matrix must be a list of lists or a 2D NumPy array.")
    
    if n_threads <= 0:
        raise ValueError("n_threads must be a positive integer.")
    
    # Convert Matrix to NumPy array for easier handling.
    Matrix = np.array(Matrix)

    # Create an empty Composition matrix.
    n_solutions = len(Matrix)
    CompositionMatrix = np.zeros((n_solutions, n_solutions))

    # Define auxiliary function for parallel computation.
    def compute_pairwise_proportion(index):
        i, j = index
        return i, j, compute_proportion_genesolution(Matrix[i], Matrix[j])
    
    # Create combinations of index.
    index = [(i, j) for i in range(n_solutions) for j in range(i, n_solutions)]
    
    # Process in 'parallel'.
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(compute_pairwise_proportion, index))
    
    # Fill the Composition matrix.
    for i, j, proportion in results:
        CompositionMatrix[i, j] = proportion
        CompositionMatrix[j, i] = proportion  # Symmetric.
    
    return CompositionMatrix

def AmountGenes_Equals(
        Solution1: list[set], 
        Solution2: list[set]) -> np.ndarray:
    """
    AmountGenes_Equals(function): 
    Compute the matrix of shared gene counts between clusters.

    Parameters:
    - Solution1 (list[set]): Clusters of the first solution as sets.
    - Solution2 (list[set]): Clusters of the second solution as sets.

    Returns:
    - MatrixEquals (np.ndarray): Matrix representing the number of shared genes between clusters.
    """
    # Create an empty matrix to store shared gene counts.
    n = len(Solution1)
    MatrixEquals = np.zeros((n, n))

    # Compare clusters from both solutions.
    for i in range(len(Solution1)):
        for j in range(i, len(Solution2)):
            intersection = len(Solution1[i] & Solution2[j])  # Intersection of sets.
            
            MatrixEquals[i, j] = intersection
            MatrixEquals[j, i] = intersection

    return MatrixEquals
