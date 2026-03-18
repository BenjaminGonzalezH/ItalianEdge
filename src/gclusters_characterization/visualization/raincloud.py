"""
raincloud.py

This module provides RainCloud-style visualization for similarity distributions.

Functions:
1. plot_similarity_raincloud_html:
   Generates an interactive RainCloud visualization from numeric data.

2. _build_raincloud_figure:
   Constructs the Plotly figure with distribution, boxplot, and dotplot.

3. _validate_input:
   Validates numeric input data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging


logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


# ─────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class RaincloudOptions:
    """
    Configuration options for RainCloud visualization.

    Attributes
    ----------
    jitter_strength : float
        Horizontal spread of points.
    point_size : int
        Size of scatter points.
    violin_width : float
        Width of violin plot.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        Enable logging.
    """

    jitter_strength: float = 0.15
    point_size: int = 6
    violin_width: float = 0.8
    random_state: int = 42
    verbose: bool = True


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _log(msg: str, verbose: bool):
    """
    Log message if verbose is enabled.
    """
    if verbose:
        logger.info(msg)


def _validate_input(values: np.ndarray):
    """
    Validate numeric input array.

    Parameters
    ----------
    values : np.ndarray
        Input numeric array.

    Raises
    ------
    TypeError, ValueError
    """
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be numpy.ndarray")

    if values.ndim != 1:
        raise ValueError("values must be 1D")

    if len(values) == 0:
        raise ValueError("values cannot be empty")

    if not np.isfinite(values).all():
        raise ValueError("values contain NaN or Inf")


# ─────────────────────────────────────────────
# Plot construction
# ─────────────────────────────────────────────

def _build_raincloud_figure(
    values: np.ndarray,
    label: str,
    options: RaincloudOptions
):
    """
    Build RainCloud Plotly figure.

    Parameters
    ----------
    values : np.ndarray
        Numeric values to visualize.
    label : str
        Label for plot.
    options : RaincloudOptions
        Visualization options.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    rng = np.random.default_rng(options.random_state)

    jitter = rng.uniform(
        -options.jitter_strength,
        options.jitter_strength,
        size=len(values)
    )

    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.4, 0.3, 0.3],
        subplot_titles=["Distribution", "Box statistics", "Observations"]
    )

    # Distribution (violin)
    fig.add_trace(
        go.Violin(
            y=values,
            name=label,
            box_visible=False,
            meanline_visible=True,
            width=options.violin_width,
        ),
        row=1,
        col=1
    )

    # Boxplot
    fig.add_trace(
        go.Box(
            y=values,
            name=label,
            boxmean=True
        ),
        row=1,
        col=2
    )

    # Dotplot
    fig.add_trace(
        go.Scatter(
            x=jitter,
            y=values,
            mode="markers",
            marker=dict(size=options.point_size),
            name="points"
        ),
        row=1,
        col=3
    )

    fig.update_layout(
        title=f"RainCloud Distribution: {label}",
        showlegend=False,
        template="plotly_white"
    )

    return fig


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def plot_similarity_raincloud_html(
    dataframe: pd.DataFrame,
    column: str,
    *,
    label: Optional[str] = None,
    options: RaincloudOptions = RaincloudOptions(),
    save_html_to: Optional[PathLike] = None,
    return_fig: bool = False,
    return_html: bool = False
):
    """
    Generate a RainCloud visualization from a dataframe column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input data containing similarity values.
    column : str
        Column to visualize.
    label : str, optional
        Label used in the plot.
    options : RaincloudOptions
        Visualization configuration.
    save_html_to : PathLike, optional
        Path to save HTML output.
    return_fig : bool
        Whether to return Plotly figure.
    return_html : bool
        Whether to return HTML string.

    Returns
    -------
    Optional[Union[Figure, str, Tuple]]
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas.DataFrame")

    if column not in dataframe.columns:
        raise ValueError(f"Column {column} not found")

    series = dataframe[column]

    # Convert to numeric safely
    numeric_values = pd.to_numeric(series, errors="coerce")

    if numeric_values.isna().all():
        raise ValueError("No valid numeric values after conversion")


    values = numeric_values.dropna().to_numpy(dtype=float)

    _validate_input(values)

    if label is None:
        label = column

    fig = _build_raincloud_figure(values, label, options)

    html = None

    if save_html_to or return_html:
        html = fig.to_html(include_plotlyjs="cdn", full_html=True)

        if save_html_to:
            p = Path(save_html_to)

            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)

            p.write_text(html, encoding="utf-8")
            _log(f"[RainCloud] HTML saved at: {p}", options.verbose)

    if return_fig and return_html:
        return fig, html
    if return_fig:
        return fig
    if return_html:
        return html

    return None