"""
Optimized SolutionClusterMatrix utilities.

Improvements:
- O(n_genes) grouping (single pass).
- Optional real parallelism using ProcessPoolExecutor.
- Optional compact output mode (indices instead of gene sets).
- Robust input validation.
"""

# ──────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import List, Set, Union, Iterable, Optional


# ──────────────────────────────────────────────────────────────
# Internal validation
# ──────────────────────────────────────────────────────────────

def _validate_inputs(matrix: np.ndarray, genes: List[str]) -> None:
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


# ──────────────────────────────────────────────────────────────
# Core grouping logic (O(n_genes))
# ──────────────────────────────────────────────────────────────

def _process_solution_sets(args) -> List[Set[str]]:
    """
    Convert one solution row into list-of-sets (gene IDs).
    Single-pass grouping: O(n_genes)
    """
    solution, genes = args
    clusters = {}

    for gene, label in zip(genes, solution):
        clusters.setdefault(label, set()).add(gene)

    return [clusters[k] for k in sorted(clusters)]


def _process_solution_indices(solution: np.ndarray) -> List[np.ndarray]:
    """
    Convert one solution row into list-of-index arrays.
    More RAM-efficient than sets.
    """
    clusters = {}
    for idx, label in enumerate(solution):
        clusters.setdefault(label, []).append(idx)

    # Convert lists to numpy arrays for efficiency
    return [np.array(indices, dtype=np.int32) for indices in clusters.values()]


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def solution_cluster_matrix(
    matrix: np.ndarray,
    genes: List[str],
    *,
    mode: str = "sets",
    parallel: bool = False,
    max_workers: Optional[int] = None,
) -> List:
    """
    Generate clustered representation for each solution.

    Parameters
    ----------
    matrix : np.ndarray
        Shape (n_solutions, n_genes)
    genes : list[str]
        Gene identifiers.
    mode : str
        "sets"   → returns List[List[Set[str]]]
        "indices" → returns List[List[np.ndarray]] (RAM efficient)
    parallel : bool
        If True, uses ProcessPoolExecutor (real parallelism).
    max_workers : int or None
        Number of processes (default: OS decides).

    Returns
    -------
    List of clustered solutions.
    """

    _validate_inputs(matrix, genes)

    if mode not in {"sets", "indices"}:
        raise ValueError("mode must be 'sets' or 'indices'.")

    n_solutions = matrix.shape[0]

    # ─────────────────────────────
    # Non-parallel mode (often fastest for small/medium datasets)
    # ─────────────────────────────
    if not parallel:

        if mode == "sets":
            return [
                _process_solution_sets((matrix[i], genes))
                for i in range(n_solutions)
            ]

        else:  # indices mode
            return [
                _process_solution_indices(matrix[i])
                for i in range(n_solutions)
            ]

    # ─────────────────────────────
    # Parallel mode (true CPU parallelism)
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

        else:  # indices mode
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    executor.map(
                        _process_solution_indices,
                        (matrix[i] for i in range(n_solutions))
                    )
                )
            return results
