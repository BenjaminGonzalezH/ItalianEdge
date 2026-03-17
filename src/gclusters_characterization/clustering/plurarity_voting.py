"""
plurality_voting.py

Implementation of a Plurality Voting ensemble clustering strategy with
consensus confidence visualization.

This module receives multiple clustering solutions and produces a
consensus clustering by selecting the most frequent label per element
after aligning cluster labels across partitions.

Functionality
1. Validate input clustering solutions matrix.
2. Align cluster labels across partitions.
3. Compute consensus clustering using plurality voting.
4. Estimate confidence (proportion of votes) for each element.
5. Optionally visualize confidence using an interactive HTML plot.

Functions
1. _validate_solutions_matrix
2. _align_partition
3. _plurality_vote
4. plot_consensus_confidence
5. plurality_voting
"""

import numpy as np
import plotly.graph_objects as go

from dataclasses import dataclass
from typing import Tuple, Optional

from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class PVOptions:
    """
    Configuration parameters for the plurality voting ensemble algorithm.

    Parameters
    ----------
    random_state : int
        Random seed reserved for potential reproducibility extensions.
    """

    random_state: int = 42


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
    Compute consensus labels and confidence using plurality voting.

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
    labels : numpy.ndarray
        Consensus cluster labels.

    confidence : numpy.ndarray
        Proportion of votes supporting the selected label.
    """

    n_solutions, n_elements = matrix.shape

    labels_out = np.zeros(n_elements, dtype=int)
    confidence = np.zeros(n_elements, dtype=float)

    for i in range(n_elements):

        column = matrix[:, i]

        labels, counts = np.unique(column, return_counts=True)

        max_count = counts.max()

        winners = labels[counts == max_count]

        chosen = winners.min()

        labels_out[i] = chosen
        confidence[i] = max_count / n_solutions

    return labels_out, confidence


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_consensus_confidence(
    confidence: np.ndarray,
    *,
    save_html_to: Optional[str] = None,
    return_html: bool = False,
    return_fig: bool = False
):
    """
    Generate an interactive HTML visualization of consensus confidence.

    Each bar represents the proportion of votes supporting the selected
    label for each element.

    Parameters
    ----------
    confidence : numpy.ndarray
        Confidence values per element.

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

    indices = list(range(len(confidence)))

    fig = go.Figure()

    fig.add_bar(
        x=indices,
        y=confidence,
        marker_color="steelblue",
        hovertemplate="Element %{x}<br>Confidence %{y:.2f}<extra></extra>"
    )

    fig.update_layout(
        title="Consensus Confidence (Plurality Voting Strength)",
        xaxis_title="Element index",
        yaxis_title="Proportion of votes",
        yaxis=dict(range=[0, 1]),
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
    plot_confidence: bool = False,
    save_plot_to: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a consensus clustering using plurality voting.

    Workflow
    --------
    1. Validate clustering solutions matrix.
    2. Select the first partition as reference.
    3. Align all partitions to the reference partition.
    4. Apply plurality voting to produce the consensus clustering.
    5. Compute confidence for each element.

    Parameters
    ----------
    solutions_matrix : numpy.ndarray
        Matrix of clustering solutions (n_partitions × n_elements).

    options : PVOptions
        Configuration options.

    plot_confidence : bool
        If True, generate a confidence visualization.

    save_plot_to : str, optional
        File path where the HTML plot will be saved.

    Returns
    -------
    tuple
        consensus : numpy.ndarray
            Final consensus clustering.

        confidence : numpy.ndarray
            Confidence score per element.
    """

    _validate_solutions_matrix(solutions_matrix)

    reference = solutions_matrix[0]

    aligned = []

    for part in solutions_matrix:

        if np.array_equal(part, reference):
            aligned.append(part)

        else:
            aligned.append(
                _align_partition(reference, part)
            )

    aligned_matrix = np.vstack(aligned)

    consensus, confidence = _plurality_vote(aligned_matrix)

    if plot_confidence:

        plot_consensus_confidence(
            confidence,
            save_html_to=save_plot_to
        )

    return consensus, confidence