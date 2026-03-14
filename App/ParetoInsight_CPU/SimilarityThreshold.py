"""
SimilarityThreshold.py

Utilities for estimating similarity thresholds using Gaussian Mixture Models.

Features
--------
- Fit Gaussian Mixture Models to similarity distributions.
- Compute thresholds using Gaussian intersections.
- Support for multiple Gaussian components.
- Combine two similarity metrics.
- Interactive HTML visualization (Plotly).
- Deterministic behavior with random_state.

Author: ParetoInsight
"""

from __future__ import annotations

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

    logger.info(msg)

    if verbose:
        print(msg)


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def _validate_values(values: np.ndarray):

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

    if values1.shape != values2.shape:
        raise ValueError("Distance arrays must have same length.")

    method = options.combination_method

    if method == "mean":
        return (values1 + values2) / 2

    if method == "weighted_mean":

        w1 = options.weight_1
        w2 = options.weight_2

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

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_

    order = np.argsort(means)

    means = means[order]
    stds = stds[order]
    weights = weights[order]

    return means, stds, weights


def _compute_gaussian_intersections(means, stds, weights):

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

    means, stds, weights = _extract_gmm_parameters(gmm)

    x = np.linspace(values.min(), values.max(), 1000)

    mixture = np.zeros_like(x)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=options.bins,
            histnorm="probability density",
            opacity=0.4,
            name="data"
        )
    )

    for i in range(len(means)):

        g = weights[i] * norm.pdf(x, means[i], stds[i])

        mixture += g

        fig.add_trace(
            go.Scatter(
                x=x,
                y=g,
                mode="lines",
                name=f"G{i+1}"
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=mixture,
            mode="lines",
            line=dict(width=3, dash="dash"),
            name="mixture"
        )
    )

    for t in thresholds:

        fig.add_vline(
            x=t,
            line_dash="dot",
            line_color="black"
        )

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

    if column not in dataframe.columns:
        raise ValueError(f"{column} not found")

    values = dataframe[column].dropna().to_numpy()

    thresholds, gmm = compute_gmm_threshold(values, options)

    if plot:

        fig = _plot_gmm_html(
            values,
            thresholds,
            gmm,
            options,
            title=f"GMM Thresholds — {column}"
        )

        if save_html_to:

            p = Path(save_html_to)

            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True)

            fig.write_html(str(p))

            _log(f"[GMM] HTML saved at: {p}", options.verbose)

    return thresholds


def estimate_similarity_threshold_combined(
    dataframe: pd.DataFrame,
    columns: Tuple[str, str],
    *,
    options: GMMThresholdOptions = GMMThresholdOptions(),
    plot: bool = False,
    save_html_to: Optional[PathLike] = None
):

    col1, col2 = columns

    if col1 not in dataframe.columns or col2 not in dataframe.columns:
        raise ValueError("Specified columns not found.")

    subset = dataframe[[col1, col2]].dropna()

    values1 = subset[col1].to_numpy()
    values2 = subset[col2].to_numpy()

    combined = combine_similarity_metrics(values1, values2, options)

    dataframe["combined_similarity"] = combined

    thresholds, gmm = compute_gmm_threshold(combined, options)

    if plot:

        fig = _plot_gmm_html(
            combined,
            thresholds,
            gmm,
            options,
            title=f"GMM Thresholds — combined ({col1},{col2})"
        )

        if save_html_to:

            p = Path(save_html_to)

            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True)

            fig.write_html(str(p))

            _log(f"[GMM] HTML saved at: {p}", options.verbose)

    return thresholds