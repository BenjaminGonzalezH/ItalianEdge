"""
rand_values: High-performance utilities for computing similarity between clustering
solutions using the Rand Index (RI) and Adjusted Rand Index (ARI).


Functions
1. rand_index_solutions          – Compute pairwise Rand Index between clustering solutions.
2. adjusted_rand_index_solutions – Compute pairwise Adjusted Rand Index between clustering solutions.
3. rand_index_clusters           – Compute Rand Index similarity between clusters of two solutions.
4. adjusted_rand_index_clusters  – Compute Adjusted Rand Index similarity between clusters of two solutions.
5. compare_solutions_pair        – Match clusters between two solutions using Rand-based similarity.
6. find_equivalent_clusters_rand – Summarize cluster correspondences across multiple solutions.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
from __future__      import annotations
from dataclasses     import dataclass
from typing          import Dict, List, Set, Tuple, Literal, Any
from scipy           import sparse
from sklearn.metrics import rand_score, adjusted_rand_score
import numpy  as np
import pandas as pd

try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except Exception:
    Parallel = None
    delayed = None
    _JOBLIB_AVAILABLE = False


Metric = Literal["rand", "adjusted_rand"]


@dataclass(frozen=True)
class _EncodedSolution:
    """
    Sparse binary membership representation for one solution.

    Attributes
    ----------
    membership : scipy.sparse.csr_matrix
        Sparse matrix of shape (n_clusters, n_items).
        Entry (i, j) = 1 if item j belongs to cluster i.
    cluster_sizes : numpy.ndarray
        Size of each cluster.
    """
    membership: sparse.csr_matrix
    cluster_sizes: np.ndarray



# ──────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────

def _validate_solution_matrix(matrix: np.ndarray) -> None:
    """
    Validate a matrix of clustering labels.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix with shape (n_solutions, n_elements).

    Raises
    ------
    TypeError
        If matrix is not a NumPy array.
    ValueError
        If matrix is not valid for pairwise comparison.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Solutions_Matrix must be numpy.ndarray.")

    if matrix.ndim != 2:
        raise ValueError("Solutions_Matrix must be 2D.")

    if matrix.shape[0] == 0:
        raise ValueError("Empty solutions matrix.")

    if matrix.shape[1] < 2:
        raise ValueError("Matrix must contain at least 2 elements.")


def _validate_cluster_solutions(solutions: List[List[Set[Any]]]) -> None:
    """
    Validate cluster-based solutions.

    Parameters
    ----------
    solutions : list[list[set]]
        Collection of clustering solutions.

    Raises
    ------
    TypeError
        If the structure is invalid.
    ValueError
        If any solution is empty.
    """
    if not isinstance(solutions, list):
        raise TypeError("solutions must be a list of solutions.")

    for sol in solutions:
        if not isinstance(sol, list):
            raise TypeError("Each solution must be a list of sets.")

        if len(sol) == 0:
            raise ValueError("Each solution must contain at least one cluster.")

        for cluster in sol:
            if not isinstance(cluster, set):
                raise TypeError("Each cluster must be a set.")


def _validate_two_solutions(solution1: List[Set[Any]], solution2: List[Set[Any]]) -> None:
    """
    Validate two clustering solutions represented as list[set].

    Parameters
    ----------
    solution1 : list[set]
    solution2 : list[set]

    Raises
    ------
    TypeError
        If inputs are not list[set].
    ValueError
        If any input is empty.
    """
    if not isinstance(solution1, list) or not all(isinstance(s, set) for s in solution1):
        raise TypeError("Solution1 must be list of sets.")

    if not isinstance(solution2, list) or not all(isinstance(s, set) for s in solution2):
        raise TypeError("Solution2 must be list of sets.")

    if not solution1 or not solution2:
        raise ValueError("Solutions must not be empty.")


# ──────────────────────────────────────────────────────────────
# Cluster encoding helpers
# ──────────────────────────────────────────────────────────────


def _build_global_index(solution1: List[Set[Any]], solution2: List[Set[Any]]) -> Dict[Any, int]:
    """
    Create a stable item-to-column index for two solutions.

    Parameters
    ----------
    solution1 : list[set]
    solution2 : list[set]

    Returns
    -------
    dict
        Mapping from item to integer column index.
    """
    universe = set()
    for cluster in solution1:
        universe.update(cluster)
    for cluster in solution2:
        universe.update(cluster)

    return {item: idx for idx, item in enumerate(sorted(universe))}


def _encode_solution(solution: List[Set[Any]], item_to_col: Dict[Any, int]) -> _EncodedSolution:
    """
    Encode one solution into a sparse cluster-membership matrix.

    Parameters
    ----------
    solution : list[set]
        Clustering solution.
    item_to_col : dict
        Global item-to-column mapping.

    Returns
    -------
    _EncodedSolution
        Sparse representation of the solution.
    """
    rows: List[int] = []
    cols: List[int] = []
    data: List[int] = []
    cluster_sizes = np.empty(len(solution), dtype=np.int64)

    for i, cluster in enumerate(solution):
        cluster_sizes[i] = len(cluster)
        for item in cluster:
            rows.append(i)
            cols.append(item_to_col[item])
            data.append(1)

    membership = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(solution), len(item_to_col)),
        dtype=np.int8,
    )

    return _EncodedSolution(membership=membership, cluster_sizes=cluster_sizes)


def _pairwise_overlap_counts(
    encoded1: _EncodedSolution,
    encoded2: _EncodedSolution,
) -> np.ndarray:
    """
    Compute pairwise intersection sizes between clusters.

    Parameters
    ----------
    encoded1 : _EncodedSolution
    encoded2 : _EncodedSolution

    Returns
    -------
    numpy.ndarray
        Dense matrix where entry (i, j) is |cluster_i ∩ cluster_j|.
    """
    overlaps = (encoded1.membership @ encoded2.membership.T).toarray()
    return overlaps.astype(np.int64, copy=False)


# ──────────────────────────────────────────────────────────────
# Rand / ARI from contingency counts
# ──────────────────────────────────────────────────────────────

def _comb2(x: np.ndarray) -> np.ndarray:
    """
    Compute x choose 2 elementwise.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    x = x.astype(np.float64, copy=False)
    return x * (x - 1.0) / 2.0


def _rand_from_binary_contingency(
    n11: np.ndarray,
    n10: np.ndarray,
    n01: np.ndarray,
) -> np.ndarray:
    """
    Compute Rand Index matrix from binary contingency counts.

    Here the element-level contingency table is:

        [[n00, n01],
         [n10, n11]]

    For the current use case based on the union of both clusters,
    n00 is always zero.

    Parameters
    ----------
    n11 : numpy.ndarray
        Count of shared members.
    n10 : numpy.ndarray
        Count of elements in A but not in B.
    n01 : numpy.ndarray
        Count of elements in B but not in A.

    Returns
    -------
    numpy.ndarray
        Rand Index values.
    """
    n = n11 + n10 + n01
    total_pairs = _comb2(n)

    # Agreements
    same_same = _comb2(n11)        # correct
    diff_diff = n10 * n01          # correct

    agreements = same_same + diff_diff

    out = np.ones_like(total_pairs, dtype=np.float64)
    mask = total_pairs > 0
    out[mask] = agreements[mask] / total_pairs[mask]

    return out


def _ari_from_binary_contingency(
    n11: np.ndarray,
    n10: np.ndarray,
    n01: np.ndarray,
) -> np.ndarray:
    """
    Compute Adjusted Rand Index matrix from binary contingency counts.

    Parameters
    ----------
    n11 : numpy.ndarray
        Count of shared members.
    n10 : numpy.ndarray
        Count of elements in A but not in B.
    n01 : numpy.ndarray
        Count of elements in B but not in A.

    Returns
    -------
    numpy.ndarray
        Adjusted Rand Index values.
    """
    # Contingency matrix:
    # [[0,   n01],
    #  [n10, n11]]
    n = n11 + n10 + n01
    total_comb = _comb2(n)

    sum_comb_cells = _comb2(n11) + _comb2(n10) + _comb2(n01)

    row1 = n01
    row2 = n10 + n11
    col1 = n10
    col2 = n01 + n11

    sum_comb_rows = _comb2(row1) + _comb2(row2)
    sum_comb_cols = _comb2(col1) + _comb2(col2)

    out = np.ones_like(total_comb, dtype=np.float64)
    mask = total_comb > 0

    expected_index = np.zeros_like(total_comb, dtype=np.float64)
    expected_index[mask] = (sum_comb_rows[mask] * sum_comb_cols[mask]) / total_comb[mask]

    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
    denominator = max_index - expected_index
    numerator = sum_comb_cells - expected_index

    stable_mask = mask & (np.abs(denominator) > 1e-15)
    out[stable_mask] = numerator[stable_mask] / denominator[stable_mask]

    degenerate_mask = mask & ~stable_mask
    out[degenerate_mask] = 1.0

    return out


def _cluster_similarity_matrix_fast(
    solution1: List[Set[Any]],
    solution2: List[Set[Any]],
    metric: Metric,
) -> np.ndarray:
    """
    Fast cluster-vs-cluster similarity using sparse membership matrices.

    Parameters
    ----------
    solution1 : list[set]
    solution2 : list[set]
    metric : {"rand", "adjusted_rand"}

    Returns
    -------
    numpy.ndarray
        Similarity matrix.
    """
    item_to_col = _build_global_index(solution1, solution2)
    encoded1 = _encode_solution(solution1, item_to_col)
    encoded2 = _encode_solution(solution2, item_to_col)

    n11 = _pairwise_overlap_counts(encoded1, encoded2)
    size1 = encoded1.cluster_sizes[:, None]
    size2 = encoded2.cluster_sizes[None, :]

    n10 = size1 - n11
    n01 = size2 - n11

    if metric == "rand":
        return _rand_from_binary_contingency(n11, n10, n01)

    if metric == "adjusted_rand":
        return _ari_from_binary_contingency(n11, n10, n01)

    raise ValueError("metric must be 'rand' or 'adjusted_rand'.")


# ──────────────────────────────────────────────────────────────
# Public API: solutions-level
# ──────────────────────────────────────────────────────────────

def _pair_score_rand(i: int, j: int, matrix: np.ndarray) -> Tuple[int, int, float]:
    """Compute one Rand score for a pair of solutions."""
    return i, j, float(rand_score(matrix[i], matrix[j]))


def _pair_score_ari(i: int, j: int, matrix: np.ndarray) -> Tuple[int, int, float]:
    """Compute one Adjusted Rand score for a pair of solutions."""
    return i, j, float(adjusted_rand_score(matrix[i], matrix[j]))


def rand_index_solutions(
    Solutions_Matrix: np.ndarray,
    *,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Compute pairwise Rand Index between clustering solutions.

    Parameters
    ----------
    Solutions_Matrix : numpy.ndarray
        Matrix of cluster labels with shape (n_solutions, n_elements).
    n_jobs : int, default=1
        Number of parallel jobs. Use -1 to use all available cores.

    Returns
    -------
    numpy.ndarray
        Symmetric similarity matrix.
    """
    _validate_solution_matrix(Solutions_Matrix)

    n_solutions = Solutions_Matrix.shape[0]
    R = np.eye(n_solutions, dtype=np.float64)

    pairs = [(i, j) for i in range(n_solutions) for j in range(i + 1, n_solutions)]

    if n_jobs != 1 and _JOBLIB_AVAILABLE and len(pairs) > 0:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_pair_score_rand)(i, j, Solutions_Matrix) for i, j in pairs
        )
    else:
        results = [_pair_score_rand(i, j, Solutions_Matrix) for i, j in pairs]

    for i, j, score in results:
        R[i, j] = score
        R[j, i] = score

    return R


def adjusted_rand_index_solutions(
    Solutions_Matrix: np.ndarray,
    *,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Compute pairwise Adjusted Rand Index between clustering solutions.

    Parameters
    ----------
    Solutions_Matrix : numpy.ndarray
        Matrix of cluster labels with shape (n_solutions, n_elements).
    n_jobs : int, default=1
        Number of parallel jobs. Use -1 to use all available cores.

    Returns
    -------
    numpy.ndarray
        Symmetric similarity matrix.
    """
    _validate_solution_matrix(Solutions_Matrix)

    n_solutions = Solutions_Matrix.shape[0]
    A = np.eye(n_solutions, dtype=np.float64)

    pairs = [(i, j) for i in range(n_solutions) for j in range(i + 1, n_solutions)]

    if n_jobs != 1 and _JOBLIB_AVAILABLE and len(pairs) > 0:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_pair_score_ari)(i, j, Solutions_Matrix) for i, j in pairs
        )
    else:
        results = [_pair_score_ari(i, j, Solutions_Matrix) for i, j in pairs]

    for i, j, score in results:
        A[i, j] = score
        A[j, i] = score

    return A


# ──────────────────────────────────────────────────────────────
# Public API: clusters-level
# ──────────────────────────────────────────────────────────────

def rand_index_clusters(Solution1: List[Set[Any]], Solution2: List[Set[Any]]) -> np.ndarray:
    """
    Compute Rand Index similarity between clusters of two solutions.

    Parameters
    ----------
    Solution1 : list[set]
        Clusters from the first solution.
    Solution2 : list[set]
        Clusters from the second solution.

    Returns
    -------
    numpy.ndarray
        Similarity matrix with shape (n_clusters_solution1, n_clusters_solution2).
    """
    _validate_two_solutions(Solution1, Solution2)
    return _cluster_similarity_matrix_fast(Solution1, Solution2, metric="rand")


def adjusted_rand_index_clusters(Solution1: List[Set[Any]], Solution2: List[Set[Any]]) -> np.ndarray:
    """
    Compute Adjusted Rand Index similarity between clusters of two solutions.

    Parameters
    ----------
    Solution1 : list[set]
        Clusters from the first solution.
    Solution2 : list[set]
        Clusters from the second solution.

    Returns
    -------
    numpy.ndarray
        Similarity matrix with shape (n_clusters_solution1, n_clusters_solution2).
    """
    _validate_two_solutions(Solution1, Solution2)
    return _cluster_similarity_matrix_fast(Solution1, Solution2, metric="adjusted_rand")


# ──────────────────────────────────────────────────────────────
# Matching utilities
# ──────────────────────────────────────────────────────────────

def compare_solutions_pair(
    idx1: int,
    idx2: int,
    solutions: List[List[Set[Any]]],
    metric: Metric = "rand",
) -> List[Tuple[int, int, float]]:
    """
    Greedily match the most similar clusters between two solutions.

    Parameters
    ----------
    idx1 : int
        Index of the first solution.
    idx2 : int
        Index of the second solution.
    solutions : list[list[set]]
        Collection of clustering solutions.
    metric : {"rand", "adjusted_rand"}, default="rand"
        Similarity metric.

    Returns
    -------
    list[tuple[int, int, float]]
        Matched cluster pairs as:
        (cluster_index_solution1, cluster_index_solution2, similarity_score)
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
    used1: Set[int] = set()
    used2: Set[int] = set()
    matches: List[Tuple[int, int, float]] = []

    flat_order = np.argsort(M, axis=None)[::-1]

    for k in flat_order:
        i, j = np.unravel_index(k, M.shape)

        if i in used1 or j in used2:
            continue

        score = float(M[i, j])
        matches.append((int(i), int(j), score))
        used1.add(int(i))
        used2.add(int(j))

        if len(used1) == n1 or len(used2) == n2:
            break

    return matches


def find_equivalent_clusters_rand(
    solutions: List[List[Set[Any]]],
    metric: Metric = "rand",
) -> pd.DataFrame:
    """
    Summarize cluster correspondences across multiple solutions.

    Parameters
    ----------
    solutions : list[list[set]]
        Collection of clustering solutions.
    metric : {"rand", "adjusted_rand"}, default="rand"
        Similarity metric used for matching.

    Returns
    -------
    pandas.DataFrame
        Summary table with columns:
        - Solution 1
        - Solution 2
        - Cluster 1
        - Cluster 2
        - Similarity
        - Metric
    """
    _validate_cluster_solutions(solutions)

    rows: List[Dict[str, Any]] = []

    for idx1 in range(len(solutions)):
        for idx2 in range(idx1 + 1, len(solutions)):
            pairs = compare_solutions_pair(idx1, idx2, solutions, metric=metric)

            for c1, c2, sim in pairs:
                rows.append(
                    {
                        "Solution 1": idx1,
                        "Solution 2": idx2,
                        "Cluster 1": c1,
                        "Cluster 2": c2,
                        f"{metric} Similarity": sim,
                    }
                )

    return pd.DataFrame(rows)