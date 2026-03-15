"""
Rand utilities for clustering solution comparison.

Features:
- rand_index_solutions: Pairwise Rand Index matrix between solutions (RAM-friendly, no 3D broadcasting).
- adjusted_rand_index_solutions: Pairwise Adjusted Rand Index (ARI) matrix between solutions.
- rand_index_clusters: Pairwise Rand Index matrix between clusters (binary membership over union universe).
- adjusted_rand_index_clusters: Pairwise ARI matrix between clusters (binary membership).
- compare_solutions_pair: Greedy best-matching of clusters between two solutions using RI or ARI.
- find_equivalent_clusters_rand: Summary DataFrame across all solution pairs.

Backwards compatibility:
- RandIndexSolutions(Solutions_Matrix, n_threads) wrapper is provided (n_threads ignored).
"""

# ──────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Set, Tuple, Literal, Iterable, Optional

Metric = Literal["rand", "adjusted_rand"]


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _validate_solution_matrix(matrix: np.ndarray) -> None:
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Solutions_Matrix must be numpy.ndarray.")
    if matrix.ndim != 2:
        raise ValueError("Solutions_Matrix must be 2D.")
    if matrix.shape[0] == 0:
        raise ValueError("Empty solutions matrix.")
    if matrix.shape[1] < 2:
        raise ValueError("Matrix must contain at least 2 genes.")


def _upper_triangle_pairs(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def _solution_to_pair_vector(solution: np.ndarray, tri: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Boolean vector over upper-triangular gene pairs.
    True means the pair is in the same cluster within this solution.
    """
    i, j = tri
    return solution[i] == solution[j]


def _comb2(x: np.ndarray) -> np.ndarray:
    """
    Compute nC2 for array-like n values.
    Uses integer-safe formula: n*(n-1)//2
    """
    x = np.asarray(x, dtype=np.int64)
    return (x * (x - 1)) // 2


def _contingency_counts(labels_a: np.ndarray, labels_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build contingency counts n_ij between two cluster labelings.

    Returns:
        nij: flattened contingency counts (only non-zero bins if you want, but we keep full bins in flattened form)
        a_sum: counts per cluster in A
        b_sum: counts per cluster in B
        n: number of items
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape != labels_b.shape:
        raise ValueError("Both label vectors must have the same shape.")
    n = int(labels_a.size)
    if n == 0:
        raise ValueError("Label vectors must be non-empty.")

    # Factorize to compact 0..k-1 ids (works for strings, ints, etc.)
    _, a_inv = np.unique(labels_a, return_inverse=True)
    _, b_inv = np.unique(labels_b, return_inverse=True)

    k_a = int(a_inv.max()) + 1
    k_b = int(b_inv.max()) + 1

    # 2D bincount by linearizing (a,b) pairs into one code
    codes = a_inv.astype(np.int64) * k_b + b_inv.astype(np.int64)
    nij = np.bincount(codes, minlength=k_a * k_b)

    a_sum = np.bincount(a_inv, minlength=k_a)
    b_sum = np.bincount(b_inv, minlength=k_b)

    return nij, a_sum, b_sum, n


def _adjusted_rand_from_counts(nij: np.ndarray, a_sum: np.ndarray, b_sum: np.ndarray, n: int) -> float:
    """
    Compute Adjusted Rand Index (ARI) from contingency counts.

    ARI = (Index - Expected) / (Max - Expected)
    where
      Index    = sum_ij C(nij, 2)
      Expected = sum_i C(ai,2) * sum_j C(bj,2) / C(n,2)
      Max      = 0.5 * (sum_i C(ai,2) + sum_j C(bj,2))
    """
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 0.0

    sum_nij = int(_comb2(nij).sum())
    sum_a = int(_comb2(a_sum).sum())
    sum_b = int(_comb2(b_sum).sum())

    expected = (sum_a * sum_b) / total_pairs
    max_index = 0.5 * (sum_a + sum_b)

    denom = max_index - expected
    if denom == 0:
        return 0.0

    return float((sum_nij - expected) / denom)


def _rand_index_from_pair_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Rand Index over pair-vectors:
      TP = same in both
      TN = different in both
      RI = (TP + TN) / total_pairs
    """
    # v1/v2 are boolean arrays of length = n_pairs
    tp = int(np.sum(v1 & v2))
    tn = int(np.sum((~v1) & (~v2)))
    total = int(v1.size)
    return float((tp + tn) / total) if total > 0 else 0.0


def _validate_cluster_solutions(solutions: List[List[Set]]) -> None:
    if not isinstance(solutions, list) or not all(isinstance(sol, list) for sol in solutions):
        raise TypeError("Each solution must be a list of sets.")
    for sol in solutions:
        if not all(isinstance(c, set) for c in sol):
            raise TypeError("Each cluster must be a set.")
        if len(sol) == 0:
            raise ValueError("Each solution must contain at least one cluster.")


def _binary_labels_for_cluster_pair(
    cluster_a: Set,
    cluster_b: Set,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create two binary label vectors over universe U = A ∪ B.
    label 1 = in cluster, label 0 = out of cluster
    """
    universe = list(cluster_a | cluster_b)
    # Deterministic ordering (important for tests / reproducibility)
    universe.sort()

    a = np.fromiter((1 if g in cluster_a else 0 for g in universe), dtype=np.int8)
    b = np.fromiter((1 if g in cluster_b else 0 for g in universe), dtype=np.int8)
    return a, b


# ──────────────────────────────────────────────────────────────
# Public API: solutions-level
# ──────────────────────────────────────────────────────────────

def rand_index_solutions(Solutions_Matrix: np.ndarray) -> np.ndarray:
    """
    Pairwise Rand Index matrix between clustering solutions.

    RAM-friendly implementation:
    - builds upper-triangle pair vectors per solution (size ~ n_genes*(n_genes-1)/2)
    - compares vectors without building (n_solutions, n_genes, n_genes) tensors
    """
    _validate_solution_matrix(Solutions_Matrix)

    n_solutions, n_genes = Solutions_Matrix.shape
    tri = _upper_triangle_pairs(n_genes)

    pair_vectors = [_solution_to_pair_vector(sol, tri) for sol in Solutions_Matrix]

    R = np.eye(n_solutions, dtype=float)
    for i in range(n_solutions):
        for j in range(i + 1, n_solutions):
            R[i, j] = R[j, i] = _rand_index_from_pair_vectors(pair_vectors[i], pair_vectors[j])
    return R


def adjusted_rand_index_solutions(Solutions_Matrix: np.ndarray) -> np.ndarray:
    """
    Pairwise Adjusted Rand Index (ARI) matrix between clustering solutions.

    Efficient implementation via contingency counts (no pair-vector needed).
    """
    _validate_solution_matrix(Solutions_Matrix)

    n_solutions = Solutions_Matrix.shape[0]
    A = np.eye(n_solutions, dtype=float)

    for i in range(n_solutions):
        for j in range(i + 1, n_solutions):
            nij, a_sum, b_sum, n = _contingency_counts(Solutions_Matrix[i], Solutions_Matrix[j])
            ari = _adjusted_rand_from_counts(nij, a_sum, b_sum, n)
            A[i, j] = A[j, i] = ari

    return A


# ──────────────────────────────────────────────────────────────
# Public API: clusters-level (binary membership)
# ──────────────────────────────────────────────────────────────

def rand_index_clusters(Solution1: List[Set], Solution2: List[Set]) -> np.ndarray:
    """
    Rand Index similarity matrix between clusters of two solutions.

    Each cell compares a cluster from Solution1 vs a cluster from Solution2 using
    a binary partition over the union universe U = cluster1 ∪ cluster2.
    """
    if not isinstance(Solution1, list) or not all(isinstance(s, set) for s in Solution1):
        raise TypeError("Solution1 must be list of sets.")
    if not isinstance(Solution2, list) or not all(isinstance(s, set) for s in Solution2):
        raise TypeError("Solution2 must be list of sets.")
    if not Solution1 or not Solution2:
        raise ValueError("Solutions must not be empty.")

    n1, n2 = len(Solution1), len(Solution2)
    M = np.zeros((n1, n2), dtype=float)

    for i, c1 in enumerate(Solution1):
        for j, c2 in enumerate(Solution2):
            a, b = _binary_labels_for_cluster_pair(c1, c2)
            nij, a_sum, b_sum, n = _contingency_counts(a, b)

            # Rand (non-adjusted) from contingency:
            # RI = (TP + TN) / C(n,2)
            # TP = C(n11,2) + C(n00,2)?? Wait: for binary labels:
            # same-label pairs = sum over label groups in both labelings:
            # Easiest: build pair vectors would be overkill; instead compute TP+TN via:
            # agreements = sum_ij C(nij,2)  (pairs in same group in both)  +  pairs in different groups in both
            # For binary, different-in-both can be derived using total_pairs - same_in_A - same_in_B + same_in_both
            total_pairs = n * (n - 1) // 2
            if total_pairs == 0:
                M[i, j] = 0.0
                continue

            same_in_both = int(_comb2(nij).sum())
            same_in_a = int(_comb2(a_sum).sum())
            same_in_b = int(_comb2(b_sum).sum())
            diff_in_both = total_pairs - same_in_a - same_in_b + same_in_both

            M[i, j] = float((same_in_both + diff_in_both) / total_pairs)

    return M


def adjusted_rand_index_clusters(Solution1: List[Set], Solution2: List[Set]) -> np.ndarray:
    """
    Adjusted Rand Index (ARI) matrix between clusters of two solutions,
    using binary membership over U = cluster1 ∪ cluster2 for each pair.
    """
    if not isinstance(Solution1, list) or not all(isinstance(s, set) for s in Solution1):
        raise TypeError("Solution1 must be list of sets.")
    if not isinstance(Solution2, list) or not all(isinstance(s, set) for s in Solution2):
        raise TypeError("Solution2 must be list of sets.")
    if not Solution1 or not Solution2:
        raise ValueError("Solutions must not be empty.")

    n1, n2 = len(Solution1), len(Solution2)
    M = np.zeros((n1, n2), dtype=float)

    for i, c1 in enumerate(Solution1):
        for j, c2 in enumerate(Solution2):
            a, b = _binary_labels_for_cluster_pair(c1, c2)
            nij, a_sum, b_sum, n = _contingency_counts(a, b)
            M[i, j] = _adjusted_rand_from_counts(nij, a_sum, b_sum, n)

    return M


# ──────────────────────────────────────────────────────────────
# Matching & summary (same structure as Jaccard module)
# ──────────────────────────────────────────────────────────────

def compare_solutions_pair(
    idx1: int,
    idx2: int,
    solutions: List[List[Set]],
    metric: Metric = "rand",
) -> List[Tuple[int, int, float]]:
    """
    Compare two clustering solutions (as list-of-sets clusters) and return greedy best matching pairs.

    metric:
      - "rand": uses Rand Index
      - "adjusted_rand": uses Adjusted Rand Index (ARI)
    """
    _validate_cluster_solutions(solutions)
    if not (0 <= idx1 < len(solutions)) or not (0 <= idx2 < len(solutions)):
        raise IndexError("idx1/idx2 out of range.")
    if idx1 == idx2:
        raise ValueError("idx1 and idx2 must refer to different solutions.")

    if metric == "rand":
        M = rand_index_clusters(solutions[idx1], solutions[idx2])
    elif metric == "adjusted_rand":
        M = adjusted_rand_index_clusters(solutions[idx1], solutions[idx2])
    else:
        raise ValueError("metric must be 'rand' or 'adjusted_rand'.")

    n1, n2 = M.shape
    used1, used2 = set(), set()
    matches: List[Tuple[int, int, float]] = []

    flat = np.argsort(M.ravel())[::-1]  # descending
    for k in flat:
        i = int(k // n2)
        j = int(k % n2)
        if i not in used1 and j not in used2:
            matches.append((i, j, float(M[i, j])))
            used1.add(i)
            used2.add(j)
            if len(used1) == n1 or len(used2) == n2:
                break

    return matches


def find_equivalent_clusters_rand(
    solutions: List[List[Set]],
    metric: Metric = "rand",
) -> pd.DataFrame:
    """
    Identify equivalent clusters across all solution pairs.

    Returns a DataFrame with columns:
      - Solution 1, Solution 2, Cluster 1, Cluster 2, Similarity, Metric
    """
    _validate_cluster_solutions(solutions)

    rows = []
    for idx1 in range(len(solutions)):
        for idx2 in range(idx1 + 1, len(solutions)):
            pairs = compare_solutions_pair(idx1, idx2, solutions, metric=metric)
            for c1, c2, sim in pairs:
                rows.append({
                    "Solution 1": idx1,
                    "Solution 2": idx2,
                    "Cluster 1": c1,
                    "Cluster 2": c2,
                    "Similarity": sim,
                    "Metric": metric,
                })

    return pd.DataFrame(rows)

