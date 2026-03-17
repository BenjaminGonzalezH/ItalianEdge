"""
solutioncluster_matrix.py

Utility functions for converting clustering solutions into structured
cluster representations.

Description
-----------
This module transforms a matrix of clustering solutions into grouped
cluster representations. Each solution row contains cluster labels
assigned to a set of genes. The module groups genes belonging to the
same cluster label.

The implementation focuses on performance and memory efficiency:

• Single-pass grouping algorithm (O(n_genes))
• Optional multiprocessing using ProcessPoolExecutor
• Two output formats:
    - sets of gene identifiers
    - arrays of gene indices (RAM-efficient)

Functionality
-------------
1. Validate inputs for clustering matrix and gene identifiers.
2. Convert clustering labels into grouped clusters.
3. Provide optional parallel processing.
4. Allow memory-efficient cluster representations.

Functions
---------
1. _validate_inputs
2. _process_solution_sets
3. _process_solution_indices
4. solution_cluster_matrix
"""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import List, Set, Optional


# ─────────────────────────────────────────────
# Internal validation
# ─────────────────────────────────────────────

def _validate_inputs(matrix: np.ndarray, genes: List[str]) -> None:
    """
    Validate the input clustering matrix and gene identifiers.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix of clustering solutions with shape
        (n_solutions, n_genes).

    genes : list[str]
        List of gene identifiers corresponding to matrix columns.

    Raises
    ------
    TypeError
        If matrix is not a NumPy array.

    ValueError
        If matrix dimensions are invalid or gene count mismatches.
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("Matrix must be numpy.ndarray.")

    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2D (n_solutions, n_genes).")

    if matrix.shape[0] == 0:
        raise ValueError("Matrix must contain at least one solution.")

    if matrix.shape[1] < 2:
        raise ValueError("Matrix must contain at least 2 genes.")

    if len(genes) != matrix.shape[1]:
        raise ValueError(
            f"Number of genes ({len(genes)}) does not match matrix width ({matrix.shape[1]})."
        )


# ─────────────────────────────────────────────
# Core grouping logic
# ─────────────────────────────────────────────

def _process_solution_sets(args) -> List[Set[str]]:
    """
    Convert one clustering solution into sets of gene identifiers.

    Each unique cluster label is transformed into a set containing
    the gene identifiers assigned to that cluster.

    Parameters
    ----------
    args : tuple
        Tuple containing:
        - solution (array of cluster labels)
        - genes (list of gene identifiers)

    Returns
    -------
    List[Set[str]]
        List of clusters represented as gene sets.
    """

    solution, genes = args
    clusters = {}

    for gene, label in zip(genes, solution):
        clusters.setdefault(label, set()).add(gene)

    return [clusters[k] for k in sorted(clusters)]


def _process_solution_indices(solution: np.ndarray) -> List[np.ndarray]:
    """
    Convert one clustering solution into index arrays.

    Instead of storing gene identifiers, this representation stores
    indices of the genes belonging to each cluster. This reduces memory
    consumption for large datasets.

    Parameters
    ----------
    solution : np.ndarray
        Cluster labels for a single solution.

    Returns
    -------
    List[np.ndarray]
        List of arrays containing gene indices per cluster.
    """

    clusters = {}

    for idx, label in enumerate(solution):
        clusters.setdefault(label, []).append(idx)

    return [
        np.array(indices, dtype=np.int32)
        for indices in clusters.values()
    ]


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def solution_cluster_matrix(
    matrix: np.ndarray,
    genes: List[str],
    *,
    mode: str = "sets",
    parallel: bool = False,
    max_workers: Optional[int] = None,
) -> List:
    """
    Convert clustering solutions into grouped cluster structures.

    Parameters
    ----------
    matrix : np.ndarray
        Clustering matrix with shape (n_solutions, n_genes).
        Each row represents a clustering solution.

    genes : list[str]
        Gene identifiers corresponding to matrix columns.

    mode : str
        Output format:

        "sets"
            Returns clusters as sets of gene identifiers.

        "indices"
            Returns clusters as arrays of gene indices.

    parallel : bool
        If True, execute clustering conversion in parallel.

    max_workers : int or None
        Number of worker processes used in parallel mode.

    Returns
    -------
    list
        List containing clustered representation for each solution.
    """

    _validate_inputs(matrix, genes)

    if mode not in {"sets", "indices"}:
        raise ValueError("mode must be 'sets' or 'indices'.")

    n_solutions = matrix.shape[0]

    # ─────────────────────────────
    # Serial execution
    # ─────────────────────────────

    if not parallel:

        if mode == "sets":
            return [
                _process_solution_sets((matrix[i], genes))
                for i in range(n_solutions)
            ]

        else:
            return [
                _process_solution_indices(matrix[i])
                for i in range(n_solutions)
            ]

    # ─────────────────────────────
    # Parallel execution
    # ─────────────────────────────

    else:

        if mode == "sets":

            with ProcessPoolExecutor(max_workers=max_workers) as executor:

                results = list(
                    executor.map(
                        _process_solution_sets,
                        ((matrix[i], genes) for i in range(n_solutions))
                    )
                )

            return results

        else:

            with ProcessPoolExecutor(max_workers=max_workers) as executor:

                results = list(
                    executor.map(
                        _process_solution_indices,
                        (matrix[i] for i in range(n_solutions))
                    )
                )

            return results