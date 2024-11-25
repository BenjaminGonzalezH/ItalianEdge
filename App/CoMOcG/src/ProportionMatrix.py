######### Libraries #########
import numpy as np
from scipy.sparse import csr_matrix                 # Compresed space row matrix.
from itertools import combinations
from collections import Counter

######### AUX Functions #########

def CountExperimentsPrescense(Matrix: list[list[int]]) -> list[int]:
    """
    CountExperimentsPrescense(funcion)
        Input:
            - Matrix: Represents a list of results.
        Output:
            - Counter_matrix: Counter of pair of matrix.
    """
    if not Matrix:
        raise ValueError("La matriz no puede estar vacía.")

    # Determine the number of columns dynamically
    num_columns = max(len(row) for row in Matrix)

    # Initialize a matrix to count valid entries
    count_matrix = np.zeros((num_columns, num_columns), dtype=int)

    # Iterate over all pairs of columns
    for col1, col2 in combinations(range(num_columns), 2):
        # Count rows where both columns have non-None values
        count = sum(
            1 for row in Matrix
            if col1 < len(row) and col2 < len(row) and row[col1] is not None and row[col2] is not None
        )
        # Update the matrix for both (col1, col2) and (col2, col1)
        count_matrix[col1, col2] = count
        count_matrix[col2, col1] = count  # Symmetric update
    
    # Fill diagonal with counts for individual columns
    for col in range(num_columns):
        count_matrix[col, col] = sum(
            1 for row in Matrix if col < len(row) and row[col] is not None
        )

    return count_matrix


######### Functions #########
def ProportionsMatrix(summed_matrix: csr_matrix, total_solutions: int, 
                    Matrix: list[list[int]] = [], count_prescence: int = 0) -> tuple[np.ndarray, np.ndarray] :
    """
    ProportionMatrix_Similarity(function)
        Input:
            - summed_matrix: sum of a list of sparce matrix in 
            csr format.
            - total_solutionns: Amount of solutions to study.
            - count_prescence: Flag for count for each gene his
            pressence into all experiments.

        Output:
            - Proportion_matrix: Matrix that represent the proportion
            of solutions where a pair of genes are allocates in the
            same cluster.
            - Distance_matrix: Matrix that represent the proportion
            of solutions where a pair of genes are allocates in the
            same cluster.
        Description: Proportion Matrix tell us % of cases where a pair
        of index (genes) have the same number (are in the same cluster).
        Also, % where they are not.
    """
    # Check if summed_matrix is in CSR format
    if not isinstance(summed_matrix, csr_matrix):
        raise TypeError("La matriz sumada debe ser del tipo csr_matrix.")
        
    # Check if total_solutions is valid
    if total_solutions <= 0:
        raise ValueError("El número total de soluciones debe ser mayor que cero.")
    
    try:
        if Matrix is None:
                raise ValueError("El parámetro 'Matrix' no puede ser None cuando 'count_presence' es 1.")
        
        # Output.
        if(count_prescence == 0):
            Proportion_matrix = summed_matrix.toarray() / total_solutions
        else:
            denominator_matrix = CountExperimentsPrescense(Matrix)
            with np.errstate(divide='ignore', invalid='ignore'):  # Manejo seguro de divisiones
                Proportion_matrix = summed_matrix.toarray() / denominator_matrix
                Proportion_matrix[np.isnan(Proportion_matrix)] = 0  # Reemplazar NaN con 0
                Proportion_matrix[np.isinf(Proportion_matrix)] = 0  # Reemplazar NaN con 0

        Distance_matrix = 1 - Proportion_matrix
        return Proportion_matrix, Distance_matrix

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None, None
