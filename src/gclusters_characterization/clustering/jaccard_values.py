"""
jaccard_values: Utilities for computing Jaccard similarity across clustering 
solutions. This module provides functions to compare clustering outputs at 
two levels:

1. Solution-level comparison
   Computes the Jaccard similarity between clustering solutions
   using pairwise gene co-membership.

2. Cluster-level comparison
   Computes the Jaccard similarity between clusters belonging
   to two different clustering solutions.

Functions
1. jaccard_index_solutions          – Compute similarity between clustering solutions.
2. jaccard_index_clusters           – Compute similarity between clusters of two solutions.
3. compare_solutions_pair           – Match clusters between two solutions using Jaccard similarity.
4. find_equivalent_clusters_jaccard – Identify equivalent clusters across solution sets.
5. _validate_solution_matrix        – Validate a clustering solution matrix.
6. _upper_triangle_pairs            – Return indices of the upper triangular part of a matrix.
7. _solution_to_pair_vector         – Convert a clustering solution into a pairwise membership vector.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
from typing import List, Set, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Internal Functions
# ──────────────────────────────────────────────────────────────────────────────

def _validate_solution_matrix(matrix: np.ndarray) -> None:
    """
    Validate a clustering solution matrix. The matrix must represent 
    multiple clustering solutions where rows correspond to solutions 
    and columns correspond to genes (or items).

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix of clustering labels with shape (n_solutions, n_genes).

    Raises
    ------
    TypeError
        If the input is not a numpy array.

    ValueError
        If the matrix is not 2-dimensional or contains invalid dimensions.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Solutions_Matrix must be numpy.ndarray.")
    if matrix.ndim != 2:
        raise ValueError("Solutions_Matrix must be 2D.")
    if matrix.shape[0] == 0:
        raise ValueError("Empty solutions matrix.")
    if matrix.shape[1] < 2:
        raise ValueError("Matrix must contain at least 2 genes.")


def _upper_triangle_pairs(n: int):
    """
    Return indices of the upper triangular part of a matrix. These indices 
    represent all unique pairs of elements.

    Parameters
    ----------
    n : int
        Number of elements.

    Returns
    -------
    tuple of numpy.ndarray
        Row and column indices representing pair combinations.
    """
    return np.triu_indices(n, k=1)


def _solution_to_pair_vector(solution: np.ndarray, tri_indices):
    """
    Convert a clustering solution into a pairwise membership vector. Each entry 
    indicates whether two genes belong to the same cluster.

    Parameters
    ----------
    solution : numpy.ndarray
        Array containing cluster labels for each gene.

    tri_indices : tuple
        Indices representing unique gene pairs.

    Returns
    -------
    numpy.ndarray
        Boolean vector indicating shared cluster membership.
    """
    return solution[tri_indices[0]] == solution[tri_indices[1]]


# ──────────────────────────────────────────────────────────────
#  Main Functions
# ──────────────────────────────────────────────────────────────

def jaccard_index_solutions(
    Solutions_Matrix: np.ndarray
) -> np.ndarray:
    """
    Compute the Jaccard similarity between clustering solutions. Each clustering 
    solution is converted into a vector representing pairwise gene co-membership. 
    The Jaccard index is then computed between these vectors.

    Parameters
    ----------
    Solutions_Matrix : numpy.ndarray
        Matrix of clustering labels with shape (n_solutions, n_genes).

    Returns
    -------
    numpy.ndarray
        Symmetric similarity matrix of size (n_solutions × n_solutions).
    """
    _validate_solution_matrix(Solutions_Matrix)

    n_solutions, n_genes = Solutions_Matrix.shape
    tri_indices = _upper_triangle_pairs(n_genes)

    # Precompute boolean vectors for each solution
    pair_vectors = [
        _solution_to_pair_vector(sol, tri_indices)
        for sol in Solutions_Matrix
    ]

    J = np.eye(n_solutions, dtype=float)

    for i in range(n_solutions):
        for j in range(i + 1, n_solutions):
            v1 = pair_vectors[i]
            v2 = pair_vectors[j]

            r = np.sum(v1 & v2)
            u = np.sum(v1 & ~v2)
            v = np.sum(~v1 & v2)

            denom = r + u + v
            J[i, j] = J[j, i] = r / denom if denom > 0 else 0.0

    return J

def jaccard_index_clusters(
    Solution1: List[Set],
    Solution2: List[Set]
) -> np.ndarray:
    """
    Compute the Jaccard similarity matrix between clusters. Each entry (i, j) 
    represents the similarity between cluster i from Solution1 and cluster j 
    from Solution2.

    The Jaccard similarity is defined as:

        |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    Solution1 : list of sets
        Clusters from the first solution.

    Solution2 : list of sets
        Clusters from the second solution.

    Returns
    -------
    numpy.ndarray
        Matrix of cluster similarities with shape (n_clusters1, n_clusters2).
    """
    if not isinstance(Solution1, list) or not all(isinstance(s, set) for s in Solution1):
        raise TypeError("Solution1 must be list of sets.")
    if not isinstance(Solution2, list) or not all(isinstance(s, set) for s in Solution2):
        raise TypeError("Solution2 must be list of sets.")
    if not Solution1 or not Solution2:
        raise ValueError("Solutions must not be empty.")

    n1, n2 = len(Solution1), len(Solution2)
    matrix = np.zeros((n1, n2), dtype=float)

    for i, s1 in enumerate(Solution1):
        for j, s2 in enumerate(Solution2):
            union = s1 | s2
            if union:
                matrix[i, j] = len(s1 & s2) / len(union)

    return matrix

def compare_solutions_pair(
    idx1: int,
    idx2: int,
    solutions: List[List[Set]]
) -> List[Tuple[int, int, float]]:
    """
    Identify the best matching clusters between two clustering solutions. The 
    function computes the Jaccard similarity matrix and greedily matches clusters 
    with the highest similarity while avoiding duplicate matches.

    Parameters
    ----------
    idx1 : int
        Index of the first clustering solution.

    idx2 : int
        Index of the second clustering solution.

    solutions : list of clustering solutions
        Each solution is represented as a list of sets.

    Returns
    -------
    list of tuples
        Each tuple contains:

        (cluster_index_solution1, cluster_index_solution2, similarity)
    """
    M = jaccard_index_clusters(solutions[idx1], solutions[idx2])
    n1, n2 = M.shape

    used1 = set()
    used2 = set()
    matches = []

    # Flatten indices sorted by similarity descending
    flat_indices = np.argsort(M.ravel())[::-1]

    for idx in flat_indices:
        i = idx // n2
        j = idx % n2

        if i not in used1 and j not in used2:
            matches.append((i, j, M[i, j]))
            used1.add(i)
            used2.add(j)

        if len(used1) == n1 or len(used2) == n2:
            break

    return matches

def find_equivalent_clusters_jaccard(
    solutions: List[List[Set]],
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Identify equivalent clusters across a collection of clustering solutions.

    Parameters
    ----------
    solutions : list of clustering solutions
        Each solution is represented as a list of sets.
    labels : list of str, optional
        Names for each solution. Must have the same length as ``solutions``.
        If None, integer indices are used.

    Returns
    -------
    pandas.DataFrame
        Table describing cluster matches with columns:

        - Solution 1
        - Solution 2
        - Cluster 1
        - Cluster 2
        - Jaccard Similarity
    """
    if not isinstance(solutions, list) or not all(isinstance(sol, list) for sol in solutions):
        raise TypeError("Each solution must be a list of sets.")

    if labels is not None and len(labels) != len(solutions):
        raise ValueError(
            f"labels length ({len(labels)}) must match solutions length ({len(solutions)})."
        )

    def _label(idx: int) -> str:
        return labels[idx] if labels is not None else idx

    rows = []

    for idx1 in range(len(solutions)):
        for idx2 in range(idx1 + 1, len(solutions)):
            pairs = compare_solutions_pair(idx1, idx2, solutions)

            for c1, c2, sim in pairs:
                rows.append({
                    "Solution 1": _label(idx1),
                    "Solution 2": _label(idx2),
                    "Cluster 1": c1,
                    "Cluster 2": c2,
                    "Jaccard Similarity": sim
                })

    return pd.DataFrame(rows)