######### Libraries #########
import numpy as np
import os
import itertools
from scipy.sparse import csr_matrix                 # Compresed space row matrix.
from concurrent.futures import ProcessPoolExecutor  # Process Administration.
from concurrent.futures import ThreadPoolExecutor   # Threads Administration.

######### AUX Functions #########

def compute_connectivity(Solution: list[int]) -> csr_matrix:
    """
    compute_connectivity(function)
        Input:
            - Solution: a list that allocates the number of 
            cluster where the gene is in.
        Output:
            - connectivity Matrix: Matrix that indicates pair of
            genes in the same cluster in csr_format.
        
        Description: Procedure where the process are going to
        check de solution and creates a connectivity matrix.
    """

    # Check correct type of input data (list or equivalent).
    if not isinstance(Solution, (list, tuple)):
        raise ValueError("La solución debe ser una lista o una secuencia.")
        
    # Check not empty list.
    if len(Solution) == 0:
        raise ValueError("La solución no puede estar vacía.")
        
    # Generate matches using NumPy for efficiency
    matches = np.equal.outer(Solution, Solution)
        
    # Convert matches to a CSR sparse matrix
    connectivity_matrix = csr_matrix(matches, dtype=int)
        
    return connectivity_matrix 

def process_solution_batch(SolutionsBatch: list[list[int]]) -> list[csr_matrix]:
    """
    process_solution_batch(function):
        Input:
            - SolutionsBatch: Chunk of solutions that a process is
            handling.
        Output:
            - Matrix that indicates pair of
            genes in the same cluster in csr_format.
    """
    return [compute_connectivity(solution) for solution in SolutionsBatch]

######### Functions #########

def connectivityMatrix_threads(SolutionsMatrix: list[list[int]], n_threads: int = 1) -> list[csr_matrix]:
    """
    connectivityMatrix_threads(function)
        Input: 
            - SolutionMatrix: Result from functions of reading input files
            (allocates in 'ReadSolution.py' of this package).
            - max_workers: Maximum of threads that are gonna process the
            solutions 1 by 1.
        Output:
            - connectivityMatrix: List of sparse matrix that indicates
            pair of gene who are in the same cluster in the respective
            solution.
        Description: Create a Matrix that create a square matrix with
        max length dimesion from his solution (all equals). This says
        what pair of index share the same number. Imagine the 
        following example:
           Solution 1 = [1, 2, 1, 2]

           Connectivity Matrix = [ [0,1,0,0] 
                                   [1,0,0,1]
                                   [0,0,1,0] ]
        That is for every solution in solution matrix.
    """
    # Input Check.
    if not isinstance(SolutionsMatrix, list) or not all(isinstance(x, list) for x in SolutionsMatrix):
        raise ValueError("SolutionsMatrix must be a list of lists.")
    if n_threads <= 0:
        raise ValueError("n_threads must be a positive integer.")
    
    try:
        
        # Using 'ThreadPoolExecutor' to make a parallel processing of
        # solutions.
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            connectivityMatrix = list(executor.map(compute_connectivity, SolutionsMatrix))
        
        # Output.
        return connectivityMatrix
    
    # Exceptions block.
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

def connectivityMatrix_processes(SolutionsMatrix: list[list[int]], n_jobs: int = os.cpu_count()) -> list[csr_matrix]:
    """
        connectivityMatrix_processes(function)
        Input: 
            - SolutionMatrix: Result from functions of reading input files
            (allocates in 'ReadSolution.py' of this package).
            - max_workers: Maximum of threads that are gonna process the
            solutions 1 by 1.
        Output:
            - connectivityMatrix: List of sparse matrix that indicates
            pair of gene who are in the same cluster in the respective
            solution.
        Description: Create a Matrix that create a square matrix with
        max length dimesion from his solution (all equals). This says
        what pair of index share the same number. Imagine the 
        following example:
           Solution 1 = [1, 2, 1, 2]

           Connectivity Matrix = [ [0,1,0,0] 
                                   [1,0,0,1]
                                   [0,0,1,0] ]
        That is for every solution in solution matrix.
    """
    # Input Check
    if not isinstance(SolutionsMatrix, list) or not all(isinstance(x, list) for x in SolutionsMatrix):
        raise ValueError("SolutionsMatrix must be a list of lists.")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be a positive integer.")
    
    try:
        # Divide solutions considering cores.
        chunk_size = max(1, len(SolutionsMatrix) // n_jobs)
        solution_chunks = [SolutionsMatrix[i:i + chunk_size] for i in range(0, len(SolutionsMatrix), chunk_size)]

        # Use ProcessPoolExecutor to process each chunk.
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = executor.map(process_solution_batch, solution_chunks)
        
        # unit all results.
        connectivityMatrix = list(itertools.chain.from_iterable(results))
        
        return connectivityMatrix

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None

def sum_connectivity_matrices(connectivity_matrix: list[csr_matrix]) -> csr_matrix:
    """
    sum_connectivity_matrices(funcion)
        Input:
            - connectivity_matrix: List of sparce matrix from
            'connectivityMatrix' function (csr format).
        Output:
            - summed_matrix: Sum of all connectivity matrix
            from input (csr format).
        Description: Sum all connectivity matrix in the input 
        ('connectivity_matrix' for more information).
    """    
    # Check if the list is empty
    if len(connectivity_matrix) == 0:
            raise ValueError("La lista de matrices de conectividad está vacía.")

    # Ensure all elements are CSR matrices
    if not all(isinstance(matrix, csr_matrix) for matrix in connectivity_matrix):
        raise TypeError("Todos los elementos deben ser matrices dispersas CSR.")
    
    first_shape = connectivity_matrix[0].shape
    if not all(matrix.shape == first_shape for matrix in connectivity_matrix):
        raise ValueError("Todas las matrices deben tener las mismas dimensiones.")
    
    try:
        # Sum connectivity matrix.
        summed_matrix = sum(connectivity_matrix[1:], start=connectivity_matrix[0])
        return summed_matrix
    
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None