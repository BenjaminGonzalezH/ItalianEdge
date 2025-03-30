######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from scipy.sparse import csr_matrix                 # Compresed space row matrix.
from typing import Tuple                            # Multiple returns doc.

# Note: the return is not a Tuple, this is used for
# allocates all the multiple elements of output that
# would have one function.

######### Functions #########

"""
This block contains all main functions.
"""

def ProportionsMatrix(
        summed_matrix: csr_matrix
    ) -> Tuple[list[np.ndarray], list[np.ndarray]] :
    """
    ProportionMatrix_Similarity(function): Proportion Matrix tell us % of cases where a pair
    of index (genes) have the same number (are in the same cluster). Also, % where they are not.
        
    Parameters:
    - summed_matrix (csr_matrix): Sum of a list of sparce matrix in csr format.
    - total_solutionns (int): Amount of solutions to study.

    Returns:
    - Proportion_matrix (list[ndarray]): Matrix that represent the proportion of solutions where a pair 
    of genes are allocates in the same cluster.
    - Distance_matrix (list[ndarray]): Matrix that represent the proportion of solutions where a pair 
    of genes are allocates in the same cluster.
    """
    # Check if summed_matrix is in CSR format
    if not isinstance(summed_matrix, csr_matrix):
        raise TypeError("The input matrix has to be in csr format.")
    
    try:
        with np.errstate(divide='ignore', invalid='ignore'):  # ignote invalid divitions.
            Proportion_matrix = summed_matrix.toarray() / summed_matrix[0, 0]
            Proportion_matrix[np.isnan(Proportion_matrix)] = 0  # Replace NaN with 0's
            Proportion_matrix[np.isinf(Proportion_matrix)] = 0  # Replace NaN with 0's

        Distance_matrix = 1 - Proportion_matrix
        return Proportion_matrix, Distance_matrix

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None
