######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from typing import Tuple                            # Document data type.

######### Main Functions ######### 
def consensus_matrix(
    Solutions_Matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ConsensusMatrix(function): Create a distance matrix based on the proportion where two elements,
    implicitly represented by column index, have the same value. This is for represent the proportion
    of solutions where two genes are in the same cluster.

    Parameters
    ----------
    Solutions_Matrix : np.ndarray
        Matrix of shape (n_solutions, n_genes).

    Returns
    -------
    Coincidence_Matrix : np.ndarray
        Proportion matrix of same-cluster occurrences.
    Consensus_Matrix : np.ndarray
        Distance matrix (1 - coincidence).
    """

    if not isinstance(Solutions_Matrix, np.ndarray):
        raise TypeError("Solutions_Matrix must be a numpy.ndarray.")

    if Solutions_Matrix.ndim != 2:
        raise ValueError("Solutions_Matrix must be 2D.")

    n_solutions, n_genes = Solutions_Matrix.shape

    if n_solutions == 0:
        raise ValueError("Empty solutions matrix.")
    if n_genes < 2:
        raise ValueError("Matrix must have at least 2 genes.")

    # Final coincidence accumulator
    coincidence = np.zeros((n_genes, n_genes), dtype=np.float64)

    # Iterate solution by solution (RAM friendly)
    for solution in Solutions_Matrix:

        # Group indices by cluster label
        clusters = {}
        for idx, label in enumerate(solution):
            clusters.setdefault(label, []).append(idx)

        # Update coincidence matrix
        for indices in clusters.values():
            indices = np.array(indices)
            coincidence[np.ix_(indices, indices)] += 1

    # Convert counts to proportions
    coincidence /= n_solutions

    # Diagonal must be 1
    np.fill_diagonal(coincidence, 1.0)

    consensus = 1.0 - coincidence

    return coincidence, consensus