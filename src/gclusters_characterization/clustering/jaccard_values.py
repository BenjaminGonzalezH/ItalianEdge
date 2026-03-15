"""
Jaccard utilities for clustering solution comparison.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from typing import List, Set, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Internal Functions
# ──────────────────────────────────────────────────────────────────────────────

def _validate_solution_matrix(matrix: np.ndarray) -> None:
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Solutions_Matrix must be numpy.ndarray.")
    if matrix.ndim != 2:
        raise ValueError("Solutions_Matrix must be 2D.")
    if matrix.shape[0] == 0:
        raise ValueError("Empty solutions matrix.")
    if matrix.shape[1] < 2:
        raise ValueError("Matrix must contain at least 2 genes.")


def _upper_triangle_pairs(n: int):
    return np.triu_indices(n, k=1)


def _solution_to_pair_vector(solution: np.ndarray, tri_indices):
    """Return boolean vector indicating gene pairs in same cluster."""
    return solution[tri_indices[0]] == solution[tri_indices[1]]


# ──────────────────────────────────────────────────────────────
#  Main Functions
# ──────────────────────────────────────────────────────────────

def jaccard_index_solutions(
    Solutions_Matrix: np.ndarray
) -> np.ndarray:
    """
    Compute Jaccard similarity between clustering solutions.

    Memory-efficient implementation.
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
    Compute Jaccard similarity matrix between clusters of two solutions.
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
    Compare two clustering solutions and return best matching clusters.
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
    solutions: List[List[Set]]
) -> pd.DataFrame:
    """
    Identify equivalent clusters across solution collection.
    """

    if not isinstance(solutions, list) or not all(isinstance(sol, list) for sol in solutions):
        raise TypeError("Each solution must be a list of sets.")

    rows = []

    for idx1 in range(len(solutions)):
        for idx2 in range(idx1 + 1, len(solutions)):
            pairs = compare_solutions_pair(idx1, idx2, solutions)

            for c1, c2, sim in pairs:
                rows.append({
                    "Solution 1": idx1,
                    "Solution 2": idx2,
                    "Cluster 1": c1,
                    "Cluster 2": c2,
                    "Jaccard Similarity": sim
                })

    return pd.DataFrame(rows)
