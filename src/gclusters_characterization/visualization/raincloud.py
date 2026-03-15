"""
RaincloudSimilarity.py

Interactive RainCloud style visualization for similarity distributions.

Features
--------
- Separation of distribution / boxplot / dotplot
- Interactive Plotly visualization
- HTML export
- Deterministic jitter
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

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

    jitter_strength: float = 0.15
    point_size: int = 6
    violin_width: float = 0.8
    random_state: int = 42
    verbose: bool = True


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _log(msg: str, verbose: bool):

    logger.info(msg)
    if verbose:
        print(msg)


def _validate_input(values: np.ndarray):

    if not isinstance(values, np.ndarray):
        raise TypeError("values must be numpy.ndarray")

    if values.ndim != 1:
        raise ValueError("values must be 1D")

    if len(values) == 0:
        raise ValueError("values cannot be empty")

    if not np.all(np.isfinite(values)):
        raise ValueError("values contain NaN or Inf")


# ─────────────────────────────────────────────
# Plot construction
# ─────────────────────────────────────────────

def _build_raincloud_figure(
    values: np.ndarray,
    label: str,
    options: RaincloudOptions
):

    np.random.seed(options.random_state)

    jitter = np.random.uniform(
        -options.jitter_strength,
        options.jitter_strength,
        size=len(values)
    )

    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.4, 0.3, 0.3],
        subplot_titles=[
            "Distribution",
            "Box statistics",
            "Observations"
        ]
    )

    # ─────────────────────────────
    # Violin / distribution
    # ─────────────────────────────

    fig.add_trace(
        go.Violin(
            y=values,
            name=label,
            box_visible=False,
            meanline_visible=True,
            width=options.violin_width,
            line_color="royalblue",
            fillcolor="lightblue",
            opacity=0.7,
        ),
        row=1,
        col=1
    )

    # ─────────────────────────────
    # Boxplot
    # ─────────────────────────────

    fig.add_trace(
        go.Box(
            y=values,
            name=label,
            marker_color="darkorange",
            boxmean=True
        ),
        row=1,
        col=2
    )

    # ─────────────────────────────
    # Dot plot
    # ─────────────────────────────

    fig.add_trace(
        go.Scatter(
            x=jitter,
            y=values,
            mode="markers",
            marker=dict(
                size=options.point_size,
                color="black",
                opacity=0.6
            ),
            name="points"
        ),
        row=1,
        col=3
    )

    fig.update_layout(
        title=f"Similarity RainCloud Visualization ({label})",
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

    if column not in dataframe.columns:
        raise ValueError(f"Column {column} not found")

    values = dataframe[column].dropna().to_numpy()

    _validate_input(values)

    if label is None:
        label = column

    fig = _build_raincloud_figure(values, label, options)

    html = None

    if save_html_to or return_html:

        html = fig.to_html(
            include_plotlyjs="cdn",
            full_html=True
        )

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