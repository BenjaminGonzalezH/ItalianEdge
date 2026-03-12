"""
SimilarityThreshold.py

Utilities for estimating similarity thresholds using Gaussian Mixture Models.

Features
--------
- Fit Gaussian Mixture Models to similarity distributions.
- Compute threshold using Gaussian intersection.
- Optional visualization.
- Optional PNG export.
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
import matplotlib
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import brentq


matplotlib.use("Agg")

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# ──────────────────────────────────────────────────────────────
# Options
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GMMThresholdOptions:
    """
    Configuration options for GMM threshold estimation.
    """

    n_components: int = 2
    random_state: int = 0
    bins: int = 50
    verbose: bool = True


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────

def _validate_values(values: np.ndarray) -> None:
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be numpy.ndarray")

    if values.ndim != 1:
        raise ValueError("values must be 1D")

    if len(values) < 10:
        raise ValueError("Too few values for GMM estimation.")

    if not np.all(np.isfinite(values)):
        raise ValueError("values contain NaN or Inf.")


def _log(msg: str, verbose: bool):
    logger.info(msg)
    if verbose:
        print(msg)


# ──────────────────────────────────────────────────────────────
# Core Computation
# ──────────────────────────────────────────────────────────────

def compute_gmm_threshold(
    values: np.ndarray,
    options: GMMThresholdOptions = GMMThresholdOptions(),
) -> Tuple[float, GaussianMixture]:
    """
    Estimate similarity threshold using Gaussian Mixture Model intersection.

    Parameters
    ----------
    values
        1D similarity array.
    options
        GMMThresholdOptions

    Returns
    -------
    threshold
        Estimated similarity threshold.
    model
        Fitted GaussianMixture model.
    """

    _validate_values(values)

    X = values.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=options.n_components,
        random_state=options.random_state,
    )

    gmm.fit(X)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_

    order = np.argsort(means)

    means = means[order]
    stds = stds[order]
    weights = weights[order]

    def gaussian_intersection(x):

        g1 = weights[0] * norm.pdf(x, means[0], stds[0])
        g2 = weights[1] * norm.pdf(x, means[1], stds[1])

        return g1 - g2

    threshold = brentq(gaussian_intersection, means[0], means[1])

    _log(f"[GMM] Threshold estimated: {threshold:.4f}", options.verbose)

    return threshold, gmm


# ──────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────

def plot_gmm_threshold(
    values: np.ndarray,
    threshold: float,
    gmm: GaussianMixture,
    *,
    options: GMMThresholdOptions = GMMThresholdOptions(),
    save_png_to: Optional[PathLike] = None,
    return_fig: bool = False,
):
    """
    Plot GMM threshold estimation.

    Parameters
    ----------
    values
        Similarity values.
    threshold
        Estimated threshold.
    gmm
        Fitted GaussianMixture model.
    """

    X = values.reshape(-1, 1)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_

    order = np.argsort(means)

    means = means[order]
    stds = stds[order]
    weights = weights[order]

    x = np.linspace(values.min(), values.max(), 1000)

    g1 = weights[0] * norm.pdf(x, means[0], stds[0])
    g2 = weights[1] * norm.pdf(x, means[1], stds[1])

    fig, ax = plt.subplots()

    ax.hist(values, bins=options.bins, density=True, alpha=0.5)

    ax.plot(x, g1, label="Gaussian 1")
    ax.plot(x, g2, label="Gaussian 2")

    ax.axvline(threshold, color="red", linestyle="--", label="Threshold")

    ax.set_title("GMM similarity threshold")
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Density")

    ax.legend()

    if save_png_to:

        p = Path(save_png_to)

        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(p)

        _log(f"[GMM] Figure saved at: {p}", options.verbose)

    if return_fig:
        return fig

    plt.close(fig)

    return None


# ──────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────

def estimate_similarity_threshold(
    dataframe: pd.DataFrame,
    column: str,
    *,
    options: GMMThresholdOptions = GMMThresholdOptions(),
    plot: bool = False,
    save_png_to: Optional[PathLike] = None,
):
    """
    High-level helper for threshold estimation directly from DataFrame.
    """

    if column not in dataframe.columns:
        raise ValueError(f"Column {column} not found.")

    values = dataframe[column].dropna().to_numpy()

    threshold, gmm = compute_gmm_threshold(values, options)

    if plot:
        plot_gmm_threshold(
            values,
            threshold,
            gmm,
            options=options,
            save_png_to=save_png_to,
        )

    return threshold