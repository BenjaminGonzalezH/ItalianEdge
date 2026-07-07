"""
similarity_threshold.py

Gaussian Mixture Model (GMM)-based threshold estimation for similarity data.

Description
-----------
This module estimates thresholds in similarity distributions using Gaussian
Mixture Models (GMM). It is designed to identify separation points between
different similarity regimes (e.g., low vs high similarity).

Core idea:
----------
A similarity distribution is modeled as a mixture of Gaussian components.
Thresholds are obtained as intersection points between adjacent Gaussians.

This approach is commonly used to:
• separate signal vs noise
• identify similarity cutoffs
• detect natural clusters in similarity space

Functionality
-------------
1. Validate similarity input values.
2. Fit a Gaussian Mixture Model (GMM).
3. Extract Gaussian parameters (means, variances, weights).
4. Compute intersections between Gaussian components.
5. Optionally combine multiple similarity metrics.
6. Provide interactive HTML visualization.

Functions
---------
1. combine_similarity_metrics
2. compute_gmm_threshold
3. estimate_similarity_threshold
4. estimate_similarity_threshold_combined
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import brentq


logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


# ─────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class GMMThresholdOptions:
    """
    Configuration for GMM-based threshold estimation.

    Parameters
    ----------
    n_components : int
        Number of Gaussian components to fit.

    random_state : int
        Random seed for reproducibility.

    bins : int
        Number of bins for histogram visualization.

    combination_method : str
        Method used to combine two similarity metrics.

    weight_1, weight_2 : float
        Weights used for weighted combinations.

    verbose : bool
        If True, print progress messages.
    """

    n_components: int = 2
    random_state: int = 0
    bins: int = 50

    combination_method: str = "mean"
    weight_1: float = 0.5
    weight_2: float = 0.5

    verbose: bool = True


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def _log(msg: str, verbose: bool):
    """
    Internal logging helper.

    Parameters
    ----------
    msg : str
        Message to log.

    verbose : bool
        If True, also prints message to console.
    """

    logger.info(msg)

    if verbose:
        print(msg)


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def _validate_values(values: np.ndarray):
    """
    Validate similarity values before GMM fitting.

    Requirements:
    • Must be 1D
    • Must contain enough samples
    • Must not contain NaN or Inf

    Parameters
    ----------
    values : np.ndarray
        Similarity values.

    Raises
    ------
    ValueError
        If validation fails.
    """

    if values.ndim != 1:
        raise ValueError("values must be 1D")

    if len(values) < 10:
        raise ValueError("Too few values for GMM estimation.")

    if not np.all(np.isfinite(values)):
        raise ValueError("values contain NaN or Inf.")


# ─────────────────────────────────────────────
# Metric combination
# ─────────────────────────────────────────────

def combine_similarity_metrics(values1, values2, options):
    """
    Combine two similarity arrays into a single metric.

    Different strategies can be used depending on the desired behavior.

    Parameters
    ----------
    values1 : np.ndarray
    values2 : np.ndarray
        Arrays of similarity values.

    options : GMMThresholdOptions
        Combination configuration.

    Returns
    -------
    np.ndarray
        Combined similarity values.
    """

    if values1.shape != values2.shape:
        raise ValueError("Distance arrays must have same length.")

    method = options.combination_method

    if method == "mean":
        return (values1 + values2) / 2

    if method == "weighted_mean":
        w1, w2 = options.weight_1, options.weight_2
        return (w1 * values1 + w2 * values2) / (w1 + w2)

    if method == "product":
        return values1 * values2

    if method == "geometric":
        return np.sqrt(values1 * values2)

    if method == "harmonic":
        return 2 * values1 * values2 / (values1 + values2 + 1e-12)

    raise ValueError("Unknown combination method")


# ─────────────────────────────────────────────
# GMM helpers
# ─────────────────────────────────────────────

def _extract_gmm_parameters(gmm):
    """
    Extract ordered Gaussian parameters from fitted GMM.

    Returns
    -------
    means : np.ndarray
    stds : np.ndarray
    weights : np.ndarray
    """

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_

    order = np.argsort(means)

    return means[order], stds[order], weights[order]


def _compute_gaussian_intersections(means, stds, weights):
    """
    Compute intersection points between adjacent Gaussian components.

    Each intersection represents a potential threshold separating
    two regions of the distribution.

    Returns
    -------
    list[float]
        Sorted list of threshold values.
    """

    intersections = []

    for i in range(len(means) - 1):

        def f(x):
            g1 = weights[i] * norm.pdf(x, means[i], stds[i])
            g2 = weights[i+1] * norm.pdf(x, means[i+1], stds[i+1])
            return g1 - g2

        try:
            x = brentq(f, means[i], means[i+1])
            intersections.append(x)

        except ValueError:
            continue

    return intersections


# ─────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────

def compute_gmm_threshold(values, options):
    """
    Fit a Gaussian Mixture Model and compute thresholds.

    Parameters
    ----------
    values : np.ndarray
        Similarity values.

    options : GMMThresholdOptions

    Returns
    -------
    thresholds : list[float]
        Intersection points between Gaussian components.

    gmm : GaussianMixture
        Fitted model.
    """

    _validate_values(values)

    X = values.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=options.n_components,
        random_state=options.random_state
    )

    gmm.fit(X)

    means, stds, weights = _extract_gmm_parameters(gmm)

    thresholds = _compute_gaussian_intersections(means, stds, weights)

    if not thresholds:
        raise RuntimeError("No Gaussian intersections found.")

    thresholds = sorted(thresholds)

    _log(f"[GMM] Thresholds: {thresholds}", options.verbose)

    return thresholds, gmm


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def _plot_gmm_html(values, thresholds, gmm, options, title):
    """
    Generate interactive Plotly visualization of GMM fit.

    Includes:
    • histogram of data
    • individual Gaussian components
    • mixture distribution
    • threshold lines
    """

    means, stds, weights = _extract_gmm_parameters(gmm)

    x = np.linspace(values.min(), values.max(), 1000)

    mixture = np.zeros_like(x)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=options.bins,
        histnorm="probability density",
        opacity=0.4,
        name="data"
    ))

    for i in range(len(means)):

        g = weights[i] * norm.pdf(x, means[i], stds[i])
        mixture += g

        fig.add_trace(go.Scatter(
            x=x,
            y=g,
            mode="lines",
            name=f"G{i+1}"
        ))

    fig.add_trace(go.Scatter(
        x=x,
        y=mixture,
        mode="lines",
        line=dict(width=3, dash="dash"),
        name="mixture"
    ))

    for t in thresholds:
        fig.add_vline(x=t, line_dash="dot", line_color="black")

    fig.update_layout(
        title=title,
        xaxis_title="Similarity",
        yaxis_title="Density",
        template="plotly_white"
    )

    return fig


# ─────────────────────────────────────────────
# High-level API
# ─────────────────────────────────────────────

def estimate_similarity_threshold(
    dataframe: pd.DataFrame,
    column: str,
    *,
    options: GMMThresholdOptions = GMMThresholdOptions(),
    plot: bool = False,
    save_html_to: Optional[PathLike] = None
):
    """
    Estimate threshold from a single similarity column.

    Parameters
    ----------
    dataframe : pd.DataFrame
    column : str
        Column containing similarity values.

    Returns
    -------
    list[float]
        Estimated thresholds.
    """

    if column not in dataframe.columns:
        raise ValueError(f"{column} not found")

    values = dataframe[column].dropna().to_numpy()

    thresholds, gmm = compute_gmm_threshold(values, options)

    if plot:

        fig = _plot_gmm_html(
            values, thresholds, gmm, options,
            title=f"GMM Thresholds — {column}"
        )

        if save_html_to:

            p = Path(save_html_to)
            p.parent.mkdir(parents=True, exist_ok=True)

            fig.write_html(str(p))
            _log(f"[GMM] HTML saved at: {p}", options.verbose)

    return thresholds


def estimate_similarity_threshold_combined(
    dataframe: pd.DataFrame,
    columns: Tuple[str, str],
    *,
    options: GMMThresholdOptions = GMMThresholdOptions(),
    plot: bool = False,
    save_html_to: Optional[PathLike] = None,
    add_combined_column: bool = False,
):
    """
    Estimate threshold from combined similarity metrics.

    Combines two columns into a single similarity measure before
    applying GMM threshold estimation.

    Parameters
    ----------
    dataframe : pd.DataFrame
    columns : tuple[str, str]
        Names of the two similarity columns to combine.
    options : GMMThresholdOptions
    plot : bool
        If True, generate an interactive GMM visualization.
    save_html_to : str or Path, optional
        File path where the HTML visualization will be saved.
    add_combined_column : bool, default False
        If True, write the combined metric back into *dataframe* under
        the column name ``"combined_similarity"``.  Set this explicitly
        to opt in to the side-effect; by default the caller's DataFrame
        is not modified.

    Returns
    -------
    list[float]
        Estimated thresholds.
    """

    col1, col2 = columns

    if col1 not in dataframe.columns or col2 not in dataframe.columns:
        raise ValueError("Specified columns not found.")

    subset = dataframe[[col1, col2]].dropna()

    values1 = subset[col1].to_numpy()
    values2 = subset[col2].to_numpy()

    combined = combine_similarity_metrics(values1, values2, options)

    if add_combined_column:
        dataframe["combined_similarity"] = combined

    thresholds, gmm = compute_gmm_threshold(combined, options)

    if plot:

        fig = _plot_gmm_html(
            combined, thresholds, gmm, options,
            title=f"GMM Thresholds — combined ({col1},{col2})"
        )

        if save_html_to:

            p = Path(save_html_to)
            p.parent.mkdir(parents=True, exist_ok=True)

            fig.write_html(str(p))
            _log(f"[GMM] HTML saved at: {p}", options.verbose)

    return thresholds

# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────
 
def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    """
    Extract the strict upper-triangle values of a square matrix (k=1).
 
    These are the n*(n-1)/2 unique pairwise values, excluding the
    diagonal (self-similarity = 1) and the redundant lower triangle.
 
    Parameters
    ----------
    matrix : np.ndarray
        Square 2-D similarity matrix.
 
    Returns
    -------
    np.ndarray
        1-D array of upper-triangle values.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a 2-D square array.")
    if matrix.shape[0] < 2:
        raise ValueError("matrix must be at least 2×2.")
 
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx].astype(np.float64)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Single-matrix API
# ──────────────────────────────────────────────────────────────────────────────
 
def estimate_similarity_threshold_from_matrix(
    matrix: np.ndarray,
    *,
    options: GMMThresholdOptions = GMMThresholdOptions(),
    metric_name: str = "Similarity",
    plot: bool = False,
    save_html_to: Optional[PathLike] = None,
) -> list:
    """
    Estimate GMM threshold from a single square similarity matrix.
 
    Parameters
    ----------
    matrix : np.ndarray
        Square 2-D similarity matrix (e.g. Jaccard or Wang).
    options : GMMThresholdOptions
        GMM configuration (n_components, combination_method, etc.).
    metric_name : str
        Label used in plot title and log messages.
    plot : bool
        If True, generate an interactive Plotly visualization.
    save_html_to : str or Path, optional
        Path where the HTML plot will be saved.
 
    Returns
    -------
    list[float]
        Estimated threshold(s).
    """
    values = _upper_triangle_values(matrix)
 
    thresholds, gmm = compute_gmm_threshold(values, options)
 
    if plot:
        fig = _plot_gmm_html(
            values, thresholds, gmm, options,
            title=f"GMM Thresholds — {metric_name}",
        )
        if save_html_to:
            p = Path(save_html_to)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(p))
            _log(f"[GMM] HTML saved at: {p}", options.verbose)
 
    return thresholds
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Dual-matrix API
# ──────────────────────────────────────────────────────────────────────────────
 
def estimate_similarity_threshold_combined_matrices(
    matrix_upper: np.ndarray,
    matrix_lower: np.ndarray,
    *,
    options: GMMThresholdOptions = GMMThresholdOptions(),
    metric_names: Tuple[str, str] = ("Jaccard", "Wang"),
    plot: bool = False,
    save_html_to: Optional[PathLike] = None,
) -> list:
    """
    Estimate GMM threshold from two square similarity matrices combined.
 
    Extracts the strict upper-triangle values from both matrices,
    combines them with ``combine_similarity_metrics`` (controlled by
    ``options.combination_method``), and fits a GMM on the result.
 
    Both matrices must be the same shape and square.  The upper-triangle
    extraction ensures each pair (i, j) with i < j is counted once,
    matching the dual-heatmap convention where matrix_upper fills the
    upper triangle and matrix_lower fills the lower triangle.
 
    Parameters
    ----------
    matrix_upper : np.ndarray
        Square similarity matrix for the first metric (e.g. Jaccard).
    matrix_lower : np.ndarray
        Square similarity matrix for the second metric (e.g. Wang).
    options : GMMThresholdOptions
        GMM + combination configuration.
        Key field: ``combination_method`` — one of:
        "mean", "weighted_mean", "product", "geometric", "harmonic".
    metric_names : tuple[str, str]
        Labels for each metric, used in plot title and log messages.
    plot : bool
        If True, generate an interactive Plotly visualization.
    save_html_to : str or Path, optional
        Path where the HTML plot will be saved.
 
    Returns
    -------
    list[float]
        Estimated threshold(s) on the combined similarity scale.
 
    Examples
    --------
    >>> thresholds = estimate_similarity_threshold_combined_matrices(
    ...     jaccard_matrix,
    ...     wang_matrix,
    ...     options=GMMThresholdOptions(
    ...         n_components=2,
    ...         combination_method="mean",
    ...     ),
    ...     plot=True,
    ...     save_html_to="gmm_dual.html",
    ... )
    """
    if matrix_upper.shape != matrix_lower.shape:
        raise ValueError("matrix_upper and matrix_lower must have the same shape.")
 
    values1 = _upper_triangle_values(matrix_upper)
    values2 = _upper_triangle_values(matrix_lower)
 
    _log(
        f"[GMM] Extracted {len(values1)} pairwise values per matrix "
        f"({metric_names[0]}, {metric_names[1]}).",
        options.verbose,
    )
 
    combined = combine_similarity_metrics(values1, values2, options)
 
    thresholds, gmm = compute_gmm_threshold(combined, options)
 
    if plot:
        label1, label2 = metric_names
        fig = _plot_gmm_html(
            combined, thresholds, gmm, options,
            title=(
                f"GMM Thresholds — combined {label1} + {label2} "
                f"({options.combination_method})"
            ),
        )
        if save_html_to:
            p = Path(save_html_to)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(p))
            _log(f"[GMM] HTML saved at: {p}", options.verbose)
 
    return thresholds