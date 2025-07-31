######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from typing import Tuple                            # Document data type.

######### AUX Functions #########

"""
This block contains all functions that are used repeatedly in other functions.
In addition, their are used by threads for process concurrency.
"""

def ConsensusMatrix(Solutions_Matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    ConsensusMatrix(function): Create a distance matrix based on the proportion where two elements,
    implicitly represented by column index, have the same value. This is for represent the proportion
    of solutions where two genes are in the same cluster.

    Parameters:
        - Solutions_Matrix: Array of clustering solutions that is, in fact, a matrix.
    
    Return:
        - Coincidence_Matrix: Square Matrix that represent the proportion of solutions where two genes 
        are in the same cluster.
        - Consensus_Matrix: Square Matrix that represent the proportion of solutions where two genes 
        are not in the same cluster. This is used for create consensus solution (heiracial clustering).
    """
    try:
        # Checking Matrix.
        if Solutions_Matrix.shape[0] == 0:
            raise ValueError("Empty solutions matrix.")
        elif Solutions_Matrix.shape[1] < 2:
            raise ValueError("Matrix at least needs to have two columns (or genes) for valid comparision in solutions.")

        # Obtain unique pairs for comparison.
        num_cols = Solutions_Matrix.shape[1]                    # Number of genes.
        upper_tri_indices = np.triu_indices(num_cols, k=1)      # Possible unique pairs of genes (columns).
        pairs = np.column_stack(upper_tri_indices)              # Format of pairs in 1D arrays of 2.

        # Broadcasting to compare each solution with the others.
        GlobalConnectivityMatrix = Solutions_Matrix[:, pairs[:, 0]] == Solutions_Matrix[:, pairs[:, 1]]

        # Sum Connectivity Matrix that are the total of cases where two genes are in the same cluster.
        Sum_ConnectivityMatrix = np.sum(GlobalConnectivityMatrix, axis=0)

        # Square Matrix copy values.
        Coincidence_Matrix = np.zeros((num_cols, num_cols), dtype=float)
        Coincidence_Matrix[pairs[:, 0], pairs[:, 1]] = Sum_ConnectivityMatrix
        Coincidence_Matrix[pairs[:, 1], pairs[:, 0]] = Sum_ConnectivityMatrix

        # Proportion calculus.
        Coincidence_Matrix = Coincidence_Matrix/Solutions_Matrix.shape[0]
        np.fill_diagonal(Coincidence_Matrix,1)                              # Fill with ones.
        Consensus_Matrix = 1 - Coincidence_Matrix                           # Distance Matrix.
        
    except ValueError as ve:
        raise RuntimeError(f"Input Matrix Error: {ve}")
    except IndexError as ie:
        raise RuntimeError(f"Index error on input matrix: {ie}")
    except Exception as e:
        raise RuntimeError(f"Something went wrong.\nDetails: {e}")
    else:
        print("Proportion Matrix & Consensus Matrix correctly calculated.")
        return Coincidence_Matrix, Consensus_Matrix