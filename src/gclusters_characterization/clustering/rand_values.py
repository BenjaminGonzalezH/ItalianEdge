"""
RandValues
----------

Utilities for computing similarity between clustering solutions
using the Rand Index (RI) and Adjusted Rand Index (ARI).

These metrics quantify how similar two clustering partitions are
by evaluating agreement in pairwise element assignments.

Two comparison levels are supported:

1. Solution-level comparison
   Measures similarity between entire clustering solutions
   (cluster assignments for all elements).

2. Cluster-level comparison
   Measures similarity between individual clusters using
   binary membership vectors defined over the union of elements.

Additional utilities are included for:

• matching clusters between two clustering solutions
• identifying equivalent clusters across multiple solutions

Functions
---------
1. rand_index_solutions
   Compute a similarity matrix of Rand Index values between solutions.

2. adjusted_rand_index_solutions
   Compute a similarity matrix of Adjusted Rand Index values.

3. rand_index_clusters
   Compute Rand similarity between clusters of two solutions.

4. adjusted_rand_index_clusters
   Compute Adjusted Rand similarity between clusters.

5. compare_solutions_pair
   Perform greedy cluster matching between two solutions.

6. find_equivalent_clusters_rand
   Generate a summary table describing cluster correspondences.
"""

# ──────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Set, Tuple, Literal
from sklearn.metrics import rand_score, adjusted_rand_score

Metric = Literal["rand", "adjusted_rand"]


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _validate_solution_matrix(matrix: np.ndarray) -> None:
    """
    Validate the structure of a clustering solution matrix.

    The matrix represents clustering assignments where:

        rows    -> clustering solutions
        columns -> genes (or items)

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix of cluster labels.

    Raises
    ------
    TypeError
        If the input is not a NumPy array.

    ValueError
        If the matrix is not two-dimensional or contains
        invalid dimensions.
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("Solutions_Matrix must be numpy.ndarray.")

    if matrix.ndim != 2:
        raise ValueError("Solutions_Matrix must be 2D.")

    if matrix.shape[0] == 0:
        raise ValueError("Empty solutions matrix.")

    if matrix.shape[1] < 2:
        raise ValueError("Matrix must contain at least 2 genes.")


def _validate_cluster_solutions(solutions: List[List[Set]]) -> None:
    """
    Validate the structure of cluster-based solutions.

    Each clustering solution must be represented as:

        list[ set ]

    where each set corresponds to a cluster containing
    the indices of its elements.

    Example
    -------
    [
        [{0,1}, {2,3}],
        [{0,2}, {1,3}]
    ]

    Parameters
    ----------
    solutions : list[list[set]]
        Collection of clustering solutions.

    Raises
    ------
    TypeError
        If the structure does not follow list-of-lists-of-sets.

    ValueError
        If any clustering solution contains zero clusters.
    """
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
    Construct binary membership vectors for two clusters.

    The vectors are defined over the union of elements from
    both clusters:

        U = cluster_a ∪ cluster_b

    Each vector indicates whether an element belongs to
    the corresponding cluster.

    Binary encoding
    ---------------
        1 → element belongs to the cluster
        0 → element does not belong to the cluster

    Example
    -------
    cluster_a = {1,2}
    cluster_b = {2,3}

    universe = [1,2,3]

    a = [1,1,0]
    b = [0,1,1]

    These vectors can then be compared using similarity
    metrics such as Rand Index or Adjusted Rand Index.

    Parameters
    ----------
    cluster_a : set
        Elements belonging to cluster A.

    cluster_b : set
        Elements belonging to cluster B.

    Returns
    -------
    tuple of numpy.ndarray
        Binary membership vectors representing each cluster.
    """

    universe = list(cluster_a | cluster_b)

    universe.sort()

    a = np.fromiter((1 if g in cluster_a else 0 for g in universe), dtype=np.int8)
    b = np.fromiter((1 if g in cluster_b else 0 for g in universe), dtype=np.int8)

    return a, b


# ──────────────────────────────────────────────────────────────
# Public API: solutions-level
# ──────────────────────────────────────────────────────────────

def rand_index_solutions(Solutions_Matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Rand Index between clustering solutions.

    The Rand Index measures agreement between two clustering
    assignments by evaluating whether pairs of elements are
    grouped together or separated in both solutions.

    This implementation relies on the sklearn implementation
    of the Rand Index, which internally uses contingency tables
    rather than explicit pairwise comparisons.

    Parameters
    ----------
    Solutions_Matrix : numpy.ndarray
        Matrix containing cluster labels.

        Shape:
            (n_solutions, n_elements)

    Returns
    -------
    numpy.ndarray
        Symmetric similarity matrix containing Rand Index values.

        Shape:
            (n_solutions × n_solutions)
    """

    _validate_solution_matrix(Solutions_Matrix)

    n_solutions = Solutions_Matrix.shape[0]

    R = np.eye(n_solutions)

    for i in range(n_solutions):
        for j in range(i + 1, n_solutions):

            score = rand_score(
                Solutions_Matrix[i],
                Solutions_Matrix[j],
            )

            R[i, j] = R[j, i] = score

    return R


def adjusted_rand_index_solutions(Solutions_Matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Adjusted Rand Index (ARI) between clustering solutions.

    The Adjusted Rand Index corrects the Rand Index by accounting
    for the similarity expected due to random cluster assignments.

    ARI values range between:

        -1 → completely different partitions
         0 → random similarity
         1 → identical partitions

    This implementation uses sklearn.metrics.adjusted_rand_score.

    Parameters
    ----------
    Solutions_Matrix : numpy.ndarray
        Matrix of clustering labels.

    Returns
    -------
    numpy.ndarray
        Symmetric similarity matrix containing ARI values.
    """

    _validate_solution_matrix(Solutions_Matrix)

    n_solutions = Solutions_Matrix.shape[0]

    A = np.eye(n_solutions)

    for i in range(n_solutions):
        for j in range(i + 1, n_solutions):

            score = adjusted_rand_score(
                Solutions_Matrix[i],
                Solutions_Matrix[j],
            )

            A[i, j] = A[j, i] = score

    return A


# ──────────────────────────────────────────────────────────────
# Public API: clusters-level
# ──────────────────────────────────────────────────────────────

def rand_index_clusters(Solution1: List[Set], Solution2: List[Set]) -> np.ndarray:
    """
    Compute Rand Index similarity between clusters of two solutions.

    Each cluster pair is converted into binary membership vectors
    over the union of elements:

        U = cluster_1 ∪ cluster_2

    These vectors are then compared using the sklearn Rand Index.

    Parameters
    ----------
    Solution1 : list[set]
        Clusters belonging to the first solution.

    Solution2 : list[set]
        Clusters belonging to the second solution.

    Returns
    -------
    numpy.ndarray
        Similarity matrix with shape:

            (n_clusters_solution1 × n_clusters_solution2)
    """

    if not isinstance(Solution1, list) or not all(isinstance(s, set) for s in Solution1):
        raise TypeError("Solution1 must be list of sets.")

    if not isinstance(Solution2, list) or not all(isinstance(s, set) for s in Solution2):
        raise TypeError("Solution2 must be list of sets.")

    if not Solution1 or not Solution2:
        raise ValueError("Solutions must not be empty.")

    n1, n2 = len(Solution1), len(Solution2)

    M = np.zeros((n1, n2))

    for i, c1 in enumerate(Solution1):
        for j, c2 in enumerate(Solution2):

            a, b = _binary_labels_for_cluster_pair(c1, c2)

            M[i, j] = rand_score(a, b)

    return M


def adjusted_rand_index_clusters(Solution1: List[Set], Solution2: List[Set]) -> np.ndarray:
    """
    Compute Adjusted Rand Index similarity between clusters.

    Cluster membership vectors are constructed over the union
    of cluster elements and compared using sklearn ARI.

    Parameters
    ----------
    Solution1 : list[set]
        Clusters belonging to the first solution.

    Solution2 : list[set]
        Clusters belonging to the second solution.

    Returns
    -------
    numpy.ndarray
        Matrix containing Adjusted Rand similarity values.
    """

    if not isinstance(Solution1, list) or not all(isinstance(s, set) for s in Solution1):
        raise TypeError("Solution1 must be list of sets.")

    if not isinstance(Solution2, list) or not all(isinstance(s, set) for s in Solution2):
        raise TypeError("Solution2 must be list of sets.")

    if not Solution1 or not Solution2:
        raise ValueError("Solutions must not be empty.")

    n1, n2 = len(Solution1), len(Solution2)

    M = np.zeros((n1, n2))

    for i, c1 in enumerate(Solution1):
        for j, c2 in enumerate(Solution2):

            a, b = _binary_labels_for_cluster_pair(c1, c2)

            M[i, j] = adjusted_rand_score(a, b)

    return M


def compare_solutions_pair(
    idx1: int,
    idx2: int,
    solutions: List[List[Set]],
    metric: Metric = "rand",
) -> List[Tuple[int, int, float]]:
    """
    Identify the best matching clusters between two clustering solutions.

    A similarity matrix is computed between clusters of both solutions.
    The matrix is then flattened and sorted in descending similarity.

    A greedy matching procedure is applied to select cluster pairs
    without reuse of clusters.

    Parameters
    ----------
    idx1 : int
        Index of the first clustering solution.

    idx2 : int
        Index of the second clustering solution.

    solutions : list[list[set]]
        Collection of clustering solutions.

    metric : {"rand", "adjusted_rand"}
        Similarity metric used for cluster comparison.

    Returns
    -------
    list[tuple]
        List of matched cluster pairs:

            (cluster_index_solution1,
             cluster_index_solution2,
             similarity_score)
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

    used1 = set()
    used2 = set()

    matches = []

    flat = np.argsort(M.ravel())[::-1]

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


# ──────────────────────────────────────────────────────────────
# Summary utilities
# ──────────────────────────────────────────────────────────────

def find_equivalent_clusters_rand(
    solutions: List[List[Set]],
    metric: Metric = "rand",
) -> pd.DataFrame:
    """
    Generate a summary table describing cluster correspondences
    across multiple clustering solutions.

    Each pair of solutions is compared and clusters are matched
    using the specified similarity metric.

    Parameters
    ----------
    solutions : list[list[set]]
        Collection of clustering solutions.

    metric : {"rand", "adjusted_rand"}
        Similarity metric used for cluster comparison.

    Returns
    -------
    pandas.DataFrame
        Table summarizing cluster equivalences.

        Columns
        -------
        Solution 1
        Solution 2
        Cluster 1
        Cluster 2
        Similarity
        Metric
    """

    _validate_cluster_solutions(solutions)

    rows = []

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
                        "Similarity": sim,
                        "Metric": metric,
                    }
                )

    return pd.DataFrame(rows)