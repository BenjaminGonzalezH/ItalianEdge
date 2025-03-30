######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from scipy.sparse import csr_matrix                 # Compresed space row matrix.
from concurrent.futures import ThreadPoolExecutor   # Threads Administration.

######### AUX Functions #########

"""
This block contains all functions that are used repeatedly in other functions.
In addition, their are used by threads for process concurrency.
"""

def compute_connectivity(
        Solution: np.ndarray
        ) -> csr_matrix:
    """
    compute_connectivity(function): Procedure where the process are going to
    check de solution and creates a connectivity matrix.
    
    Parameters:
    - Solutions (ndarray): Matrix with all solutions readed from respective
    function.
        
    Returns:
    - connectivity Matrix: Matrix that indicates pair of genes in the same cluster 
    in csr_format.
    """
    # Check not empty list.
    if len(Solution) == 0:
        raise ValueError("solution cannot be empty.")
        
    # Generate matches using NumPy for efficiency.
    matches = np.equal.outer(Solution, Solution)
        
    # Convert matches to a CSR sparse matrix.
    connectivity_matrix = csr_matrix(matches, dtype=int)
        
    return connectivity_matrix 

def safe_compute_connectivity(
        Solution: np.ndarray
    ) -> csr_matrix:
    """
    safe_compute_connectivity(function): Execute a exception handle 
    version of compute_connectivity function.

    Parameters:
    - Solutions (ndarray): Matrix with all solutions readed from respective
    function.

    Returns:
    - connectivity Matrix: Matrix that indicates pair of genes in the same cluster 
    in csr_format.
    """
    try:
        return compute_connectivity(Solution)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

######### Functions #########

"""
This block contains all main functions.
"""

def connectivityMatrix(
        SolutionsMatrix: list[np.ndarray], 
        n_threads: int = 1
        ) -> list[csr_matrix]:
    """
    connectivityMatrix_threads(function): Create a Matrix that create a square matrix with
    max length dimesion from his solution (all equals). This says what pair of index share 
    the same number. Imagine the following example:
    Solution 1 = [1, 2, 1, 2]

    Connectivity Matrix = [ [0,1,0,0] 
                            [1,0,0,1]
                            [0,0,1,0] ]
    That is for every solution in solution matrix.

    Parameters: 
    - SolutionMatrix (list[ndarray]): Result from functions of reading input files (allocates 
    in 'ReadSolution.py' of this package).
    - max_workers (int): Maximum of threads that are gonna process the solutions 1 by 1.
        
    Retruns:
    - connectivityMatrix (list[csr_matrix]): List of sparse matrix that indicates pair of gene 
    who are in the same cluster in the respective solution.
    """
    # Input Check.
    if isinstance(SolutionsMatrix, list):
        if not all(isinstance(row, np.ndarray) for row in SolutionsMatrix):
            raise ValueError("SolutionsMatrix has to be a list of ndarray.")
    elif isinstance(SolutionsMatrix, np.ndarray):
        if SolutionsMatrix.ndim != 2:
            raise ValueError("SolutionsMatrix has to be in 2D.")
    if n_threads <= 0:
        raise ValueError("n_threads must be a positive integer.")
    
    # Exception handling.
    try:
        # Using 'ThreadPoolExecutor' to make a parallel processing of
        # solutions.
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            connectivityMatrix = list(executor.map(safe_compute_connectivity, SolutionsMatrix))
        
        # Output.
        return connectivityMatrix
    
    # Exceptions message block.
    except Exception as e:
        print(f"Unexpected error: {e}")

def sum_connectivity_matrices(
        connectivity_matrix: list[csr_matrix]
        ) -> csr_matrix:
    """
    sum_connectivity_matrices(funcion): Sum all connectivity matrix in the input 
    ('connectivityMatrix_threads' for more information).
    
    Parameters:
    - connectivity_matrix (list[csr_matrix]): List of sparce matrix from
    'connectivityMatrix_threads' function (csr format).
    
    Returns:
    - summed_matrix (csr_matrix): Sum of all connectivity matrix from input (csr format).
    """ 
    # Input check.
    if not connectivity_matrix:
        raise ValueError("empty list.")
    
    first_shape = connectivity_matrix[0].shape
    if not all(matrix.shape == first_shape for matrix in connectivity_matrix):
        raise ValueError("All matrix needs to have the same shape.")
    
    # Exception handling.
    try:
        # Sum connectivity matrix.
        return sum(connectivity_matrix[1:], start=connectivity_matrix[0])
    
    # Exceptions message block.
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")
