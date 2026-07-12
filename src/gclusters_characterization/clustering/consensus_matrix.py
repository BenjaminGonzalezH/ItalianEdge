"""
consensus_matrix: This module computes a no_coincidence matrix from multiple clustering solutions. 
The no_coincidence matrix represents how frequently two elements (genes) are assigned to the same 
cluster across different clustering solutions.

Functions
1. no_coincidence_matrix – Compute coincidence and no_coincidence matrices from clustering solutions.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────────────────────────────────────

def consensus_matrix(
    Solutions_Matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a no_coincidence matrix from multiple clustering solutions. Each column represents an element 
    (e.g., gene), and each row represents a clustering solution assigning elements to clusters. The 
    coincidence matrix stores the proportion of solutions where two elements belong to the same cluster.

    The no_coincidence matrix is defined as: 
        no_coincidence = 1 − coincidence

    Parameters
    ----------
    Solutions_Matrix : numpy.ndarray
        Matrix of shape (n_solutions, n_genes) where each row contains
        cluster labels for a clustering solution.

    Returns
    -------
    coincidence_matrix : numpy.ndarray
        Matrix representing the proportion of times two elements appear
        in the same cluster.

    no_coincidence_matrix : numpy.ndarray
        Distance matrix computed as (1 − coincidence_matrix).

    Raises
    ------
    TypeError
        If the input is not a NumPy array.

    ValueError
        If the matrix is not 2D, empty, or contains fewer than two columns.
    """

    # Ensure input is a NumPy array.
    if not isinstance(Solutions_Matrix, np.ndarray):
        raise TypeError("Solutions_Matrix must be a numpy.ndarray.")

    # Ensure matrix has two dimensions.
    if Solutions_Matrix.ndim != 2:
        raise ValueError("Solutions_Matrix must be 2D.")

    n_solutions, n_genes = Solutions_Matrix.shape

    # Validate matrix size.
    if n_solutions == 0:
        raise ValueError("Empty solutions matrix.")

    if n_genes < 2:
        raise ValueError("Matrix must have at least 2 genes.")

    # Initialize coincidence accumulator.
    coincidence = np.zeros((n_genes, n_genes), dtype=np.float64)

    # Process each clustering solution independently.
    for solution in Solutions_Matrix:

        # Group gene indices by cluster label.
        clusters = {}

        for idx, label in enumerate(solution):
            clusters.setdefault(label, []).append(idx)

        # Update coincidence counts for each cluster.
        for indices in clusters.values():
            indices = np.array(indices)
            coincidence[np.ix_(indices, indices)] += 1

    # Convert counts to proportions.
    coincidence /= n_solutions

    # Diagonal represents self-similarity.
    np.fill_diagonal(coincidence, 1.0)

    no_coincidence = 1.0 - coincidence

    return coincidence, no_coincidence