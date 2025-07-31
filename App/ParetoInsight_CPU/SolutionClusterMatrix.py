######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.


######### Functions #########

"""
This block contains all main functions.
"""

def ProcessSolution(solution: np.ndarray, genes: list) -> list:
    """
    Processes a clustering solution by grouping genes into clusters using vectorization.

    Parameters:
    - solution: 1D NumPy array indicating the cluster index for each gene.
    - genes: List of gene identifiers.

    Returns:
    - Clusters: List of list that allocates sets representing clustered gene IDs (np.str_).
    """
    try:
        # Obtain the unique clusters labels fron solution array.
        unique_clusters = np.unique(solution)

        # Take genes that have the same cluster label from uniques and create the sets
        # according to it.
        clusters = [set(np.array(genes)[solution == cluster]) for cluster in unique_clusters]

    except TypeError as te:
        raise RuntimeError(f"Type error in input sets: {te}")
    except Exception as e:
        raise RuntimeError(f"Something went wrong: {e}")
    else:
        return clusters


def SolutionClusterMatrix(Matrix: np.ndarray, genes: list, max_workers: int = 4) -> list:
    """
    Generates a solution cluster matrix using parallel processing and vectorization.

    Parameters:
    - Matrix: 2D NumPy array with clustering solutions (rows: solutions, cols: genes).
    - genes: List of gene identifiers.
    - max_workers: Number of parallel threads.

    Returns:
    - SolutionClusterMatrix: List of clustered gene IDs for each solution.
    """
    try:
        # Concurrent execution using ThreadPoolExecutor.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            SolutionClusterMatrix = list(executor.map(ProcessSolution, Matrix, [genes]*Matrix.shape[0]))

    except Exception as e:
        raise RuntimeError(f"Something went wrong: {e}")
    else:
        return SolutionClusterMatrix