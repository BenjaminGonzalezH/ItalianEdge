"""
plurality_voting.py

Implementation of a Plurality Voting ensemble clustering strategy with
partition stability analysis and optional interactive visualization.

This module receives multiple clustering solutions and produces a
consensus clustering by selecting the most frequent label per element
after aligning cluster labels across partitions.

Functionality
1. Validate input clustering solutions matrix.
2. Compute similarity between partitions using Rand or Adjusted Rand Index.
3. Estimate the stability of each partition relative to the others.
4. Select the most stable partition as a reference.
5. Align cluster labels of all partitions to the reference partition.
6. Compute the final consensus clustering using plurality voting.
7. Optionally visualize partition stability using Plotly.

Functions
1. _validate_solutions_matrix
2. _partition_similarity
3. compute_partition_stability
4. _align_partition
5. _plurality_vote
6. plot_partition_stability
7. plurality_voting
"""

import numpy as np
import plotly.graph_objects as go

from dataclasses import dataclass
from typing import Tuple, Optional

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import rand_score, adjusted_rand_score


# ─────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class PVOptions:
    """
    Configuration parameters for the plurality voting ensemble algorithm.

    Parameters
    ----------
    similarity_metric : str
        Similarity metric used to compare clustering partitions.
        Available options:

        • "ari"  → Adjusted Rand Index (recommended)
        • "rand" → Rand Index

    random_state : int
        Random seed reserved for potential reproducibility extensions.
    """

    similarity_metric: str = "ari"
    random_state: int = 42


VALID_METRICS = {"ari", "rand"}


# ─────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────

def _validate_solutions_matrix(matrix: np.ndarray):
    """
    Validate the clustering solutions matrix.

    The expected format is a 2D matrix where:
    - rows represent clustering solutions
    - columns represent elements being clustered

    Example
    -------
    [
        [1,1,2,2],
        [1,1,2,2],
        [2,2,1,1]
    ]

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix containing clustering solutions.

    Raises
    ------
    TypeError
        If the input is not a NumPy array.

    ValueError
        If the matrix is not 2D or has insufficient dimensions.
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("Solutions matrix must be numpy.ndarray")

    if matrix.ndim != 2:
        raise ValueError("Solutions matrix must be 2D")

    n_solutions, n_elements = matrix.shape

    if n_solutions < 2:
        raise ValueError("At least two clustering solutions are required")

    if n_elements < 2:
        raise ValueError("At least two elements are required")


# ─────────────────────────────────────────────
# Partition similarity
# ─────────────────────────────────────────────

def _partition_similarity(p1, p2, metric):
    """
    Compute similarity between two clustering partitions.

    The comparison is performed element-wise using a clustering
    similarity index.

    Parameters
    ----------
    p1 : array-like
        First clustering partition.

    p2 : array-like
        Second clustering partition.

    metric : str
        Similarity metric to use:
        • "ari"  → Adjusted Rand Index
        • "rand" → Rand Index

    Returns
    -------
    float
        Similarity score between partitions.

    Raises
    ------
    ValueError
        If an unsupported metric is provided.
    """

    if metric == "ari":
        return adjusted_rand_score(p1, p2)

    if metric == "rand":
        return rand_score(p1, p2)

    raise ValueError("metric must be 'ari' or 'rand'")


# ─────────────────────────────────────────────
# Stability calculation
# ─────────────────────────────────────────────

def compute_partition_stability(
    solutions: np.ndarray,
    metric: str
) -> np.ndarray:
    """
    Compute the stability score of each partition.

    Stability is defined as the average similarity between
    a partition and all other partitions.

    A partition with higher stability is considered more
    representative of the ensemble.

    Parameters
    ----------
    solutions : numpy.ndarray
        Matrix of clustering solutions (n_partitions × n_elements).

    metric : str
        Similarity metric used for comparisons.

    Returns
    -------
    numpy.ndarray
        Stability score for each partition.
    """

    n = solutions.shape[0]
    stability = np.zeros(n)

    for i in range(n):

        sims = []

        for j in range(n):

            if i == j:
                continue

            sims.append(
                _partition_similarity(
                    solutions[i],
                    solutions[j],
                    metric
                )
            )

        stability[i] = np.mean(sims)

    return stability


# ─────────────────────────────────────────────
# Label alignment
# ─────────────────────────────────────────────

def _align_partition(reference, target):
    """
    Align cluster labels of a target partition to a reference partition.

    Clustering algorithms may assign different numeric labels to
    identical clusters. This function resolves that ambiguity by
    computing an optimal mapping between cluster labels.

    The Hungarian algorithm is used to maximize element overlap
    between clusters.

    Parameters
    ----------
    reference : array-like
        Reference clustering partition.

    target : array-like
        Partition whose labels must be aligned.

    Returns
    -------
    numpy.ndarray
        Target partition with labels remapped to match the reference.
    """

    ref_labels = np.unique(reference)
    tgt_labels = np.unique(target)

    cost = np.zeros((len(ref_labels), len(tgt_labels)))

    for i, r in enumerate(ref_labels):

        ref_mask = reference == r

        for j, t in enumerate(tgt_labels):

            tgt_mask = target == t

            cost[i, j] = -np.sum(ref_mask & tgt_mask)

    row, col = linear_sum_assignment(cost)

    mapping = {
        tgt_labels[c]: ref_labels[r]
        for r, c in zip(row, col)
    }

    return np.array([mapping.get(x, x) for x in target])


# ─────────────────────────────────────────────
# Plurality voting
# ─────────────────────────────────────────────

def _plurality_vote(matrix):
    """
    Compute consensus labels using plurality voting.

    For each element, the cluster label appearing most frequently
    across all aligned partitions is selected.

    In case of ties, the smallest label is chosen to ensure
    deterministic results.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix of aligned partitions.

    Returns
    -------
    numpy.ndarray
        Consensus cluster labels for each element.
    """

    n_solutions, n_elements = matrix.shape
    result = np.zeros(n_elements, dtype=int)

    for i in range(n_elements):

        column = matrix[:, i]

        labels, counts = np.unique(column, return_counts=True)

        winners = labels[counts == counts.max()]

        result[i] = winners.min()

    return result


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_partition_stability(
    stability: np.ndarray,
    reference_index: int,
    *,
    save_html_to: Optional[str] = None,
    return_html: bool = False,
    return_fig: bool = False
):
    """
    Generate an interactive Plotly visualization of partition stability.

    The reference partition is highlighted to indicate the solution
    chosen as the consensus baseline.

    Parameters
    ----------
    stability : numpy.ndarray
        Stability score for each partition.

    reference_index : int
        Index of the reference partition.

    save_html_to : str, optional
        File path where the HTML visualization will be saved.

    return_html : bool
        If True, return the HTML string.

    return_fig : bool
        If True, return the Plotly figure object.

    Returns
    -------
    Optional[Plotly Figure or HTML string]
    """

    if reference_index < 0 or reference_index >= len(stability):
        raise ValueError("Invalid reference_index")

    indices = list(range(len(stability)))

    colors = ["steelblue"] * len(stability)
    colors[reference_index] = "tomato"

    fig = go.Figure()

    fig.add_bar(
        x=indices,
        y=stability,
        marker_color=colors,
        hovertemplate="Partition %{x}<br>Stability %{y:.4f}<extra></extra>"
    )

    fig.add_hline(
        y=stability[reference_index],
        line_dash="dash",
        line_color="red",
        opacity=0.6
    )

    fig.update_layout(
        title="Partition Stability (Reference highlighted)",
        xaxis_title="Partition index",
        yaxis_title="Average similarity",
        template="plotly_white"
    )

    html = None

    if save_html_to or return_html:

        html = fig.to_html(include_plotlyjs="cdn", full_html=True)

        if save_html_to:
            with open(save_html_to, "w", encoding="utf-8") as f:
                f.write(html)

    if return_fig and return_html:
        return fig, html

    if return_fig:
        return fig

    if return_html:
        return html

    return None


# ─────────────────────────────────────────────
# Main algorithm
# ─────────────────────────────────────────────

def plurality_voting(
    solutions_matrix: np.ndarray,
    *,
    options: PVOptions = PVOptions(),
    plot_stability: bool = False,
    save_plot_to: Optional[str] = None
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Compute a consensus clustering using plurality voting.

    Workflow
    --------
    1. Validate clustering solutions matrix.
    2. Compute stability of each partition.
    3. Select the most stable partition as reference.
    4. Align all partitions to the reference partition.
    5. Apply plurality voting to produce the consensus clustering.

    Parameters
    ----------
    solutions_matrix : numpy.ndarray
        Matrix of clustering solutions (n_partitions × n_elements).

    options : PVOptions
        Configuration options.

    plot_stability : bool
        If True, generate a stability visualization.

    save_plot_to : str, optional
        File path where the stability plot will be saved.

    Returns
    -------
    tuple
        consensus : numpy.ndarray
            Final consensus clustering.

        reference_idx : int
            Index of the selected reference partition.

        stability : numpy.ndarray
            Stability score for each partition.
    """

    _validate_solutions_matrix(solutions_matrix)

    if options.similarity_metric not in VALID_METRICS:

        raise ValueError(
            f"Invalid metric '{options.similarity_metric}'. "
            f"Choose from {VALID_METRICS}"
        )

    stability = compute_partition_stability(
        solutions_matrix,
        options.similarity_metric
    )

    reference_idx = int(np.argmax(stability))

    reference = solutions_matrix[reference_idx]

    aligned = []

    for part in solutions_matrix:

        if np.array_equal(part, reference):
            aligned.append(part)

        else:
            aligned.append(
                _align_partition(reference, part)
            )

    aligned_matrix = np.vstack(aligned)

    consensus = _plurality_vote(aligned_matrix)

    if plot_stability:

        plot_partition_stability(
            stability,
            reference_idx,
            save_html_to=save_plot_to
        )

    return consensus, reference_idx, stability