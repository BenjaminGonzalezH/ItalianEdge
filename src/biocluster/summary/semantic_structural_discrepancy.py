"""
semantic_structural_discrepancy: Identify clustering solutions that present higher GO semantic similarity
(Wang index) than structural similarity (Jaccard index) to other solutions — or vice versa.

The core idea is that two solutions can agree biologically (shared GO terms across their clusters) while
assigning genes differently, or assign genes identically while being biologically divergent. Both cases
are detected via a discrepancy score:

    discrepancy[i, j] = wang_similarity[i, j] - jaccard_similarity[i, j]

    discrepancy >> 0  →  Wang HIGH, Jaccard LOW
                         "Semantically similar but structurally different"
                         Biologically coherent alternative solutions.

    discrepancy << 0  →  Wang LOW, Jaccard HIGH
                         "Structurally similar but semantically divergent"
                         Solutions that co-assign genes identically but
                         capture different biology.

Both matrices are assumed to be SIMILARITIES in [0, 1].
If you have Jaccard *distances* (1 - Jaccard), flip the sign before calling.

Functions
---------
compute_discrepancy_matrix
    Element-wise Wang - Jaccard, with masking of the diagonal.

compute_pairwise_discrepancy_profile
    One row per solution PAIR (i, j): wang, jaccard, discrepancy, and a
    z-score for ranking — no per-solution aggregation.

identify_discrepant_solution_pairs
    Filter solution PAIRS by z-score threshold or fixed top-K, in either
    direction (wang_over_jaccard or jaccard_over_wang).

plot_discrepancy_summary
    Two-panel interactive figure:
      · Scatter Wang vs Jaccard (per pair, coloured by discrepancy, flagged
        pairs highlighted)
      · Heatmap of the discrepancy matrix
"""

# ─────────────────────────────────────────────────────────────────────────────
# Libraries
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

# ─────────────────────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DiscrepancyOptions:
    """
    Configuration for discrepancy-based pair selection.

    Attributes
    ----------
    direction : "wang_over_jaccard" | "jaccard_over_wang"
        Which kind of discrepancy to flag.
        · "wang_over_jaccard"  →  Wang HIGH, Jaccard LOW  (default)
        · "jaccard_over_wang"  →  Jaccard HIGH, Wang LOW
    z_threshold : float
        Flag a solution pair when its discrepancy z-score (relative to the
        distribution of all pairwise discrepancies) is ≥ this value.
        Ignored when top_k is set. Default = 1.5.
    top_k : int, optional
        If given, flag exactly the top_k most discrepant solution pairs
        instead of using the z-score cutoff.
    """

    direction: Literal["wang_over_jaccard", "jaccard_over_wang"] = "wang_over_jaccard"
    z_threshold: float = 1.5
    top_k: int | None = None


def _hierarchical_order(matrix: np.ndarray) -> np.ndarray:
    """
    Devuelve el orden de índices que agrupa filas/columnas similares,
    usando clustering jerárquico sobre la matriz de discrepancia.
    """
    # Reemplazar la diagonal NaN por 0 (distancia consigo mismo)
    mat = np.nan_to_num(matrix, nan=0.0)

    # Simetrizar por si acaso (debería serlo, pero por seguridad numérica)
    mat = (mat + mat.T) / 2

    # Convertir a "distancia": valores similares deben quedar cerca
    dist = np.abs(mat)  # o (mat.max() - mat) si prefieres otra escala
    np.fill_diagonal(dist, 0)

    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    return leaves_list(z)


# ─────────────────────────────────────────────────────────────────────────────
# Core: discrepancy matrix
# ─────────────────────────────────────────────────────────────────────────────


def compute_discrepancy_matrix(
    wang_matrix: np.ndarray,
    jaccard_matrix: np.ndarray,
) -> np.ndarray:
    """
    Element-wise discrepancy: Wang similarity − Jaccard similarity.

    Parameters
    ----------
    wang_matrix : np.ndarray (n × n)
        Pairwise GO semantic similarity (Wang index). Values in [0, 1].
    jaccard_matrix : np.ndarray (n × n)
        Pairwise Jaccard similarity of gene co-assignments. Values in [0, 1].
        If you stored 1 - Jaccard (distances), negate before passing.

    Returns
    -------
    np.ndarray (n × n)
        discrepancy[i, j] = wang[i, j] - jaccard[i, j].
        Diagonal is set to NaN (self-comparison is meaningless).
    """
    wang = np.asarray(wang_matrix, dtype=np.float64)
    jacc = np.asarray(jaccard_matrix, dtype=np.float64)

    if wang.shape != jacc.shape:
        raise ValueError(f"Shape mismatch: wang {wang.shape} vs jaccard {jacc.shape}.")
    if wang.ndim != 2 or wang.shape[0] != wang.shape[1]:
        raise ValueError("Both matrices must be square (n × n).")

    disc = wang - jacc
    np.fill_diagonal(disc, np.nan)
    return disc


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair profile
# ─────────────────────────────────────────────────────────────────────────────


def compute_pairwise_discrepancy_profile(
    wang_matrix: np.ndarray,
    jaccard_matrix: np.ndarray,
    labels: Sequence[str] | None = None,
    options: DiscrepancyOptions = DiscrepancyOptions(),
) -> pd.DataFrame:
    """
    Per-pair summary of Wang-vs-Jaccard discrepancy — one row per solution
    pair (i, j), i < j. No aggregation across a solution's other pairs is
    performed.

    A signed score is derived depending on the chosen direction:
        · wang_over_jaccard  →  score = discrepancy[i, j]   (positive = Wang > Jaccard)
        · jaccard_over_wang  →  score = -discrepancy[i, j]  (positive = Jaccard > Wang)

    Parameters
    ----------
    wang_matrix, jaccard_matrix : np.ndarray (n × n)
        See ``compute_discrepancy_matrix``.
    labels : Sequence[str], optional
        Human-readable solution labels.
    options : DiscrepancyOptions
        Controls direction.

    Returns
    -------
    pd.DataFrame
        Columns: "Solution 1", "Solution 2", ["label 1", "label 2"],
        "wang", "jaccard", "discrepancy", "score", "z_score".
        Sorted descending by "score".
    """
    disc = compute_discrepancy_matrix(wang_matrix, jaccard_matrix)
    wang = np.asarray(wang_matrix, dtype=np.float64)
    jacc = np.asarray(jaccard_matrix, dtype=np.float64)
    n = disc.shape[0]

    if labels is not None and len(labels) != n:
        raise ValueError(f"labels length ({len(labels)}) ≠ n solutions ({n}).")

    rows_idx, cols_idx = np.triu_indices(n, k=1)
    disc_vals = disc[rows_idx, cols_idx]

    # Sign flip for jaccard_over_wang direction
    score = disc_vals if options.direction == "wang_over_jaccard" else -disc_vals

    std_score = score.std()
    z_score = (
        np.zeros_like(score) if std_score == 0 else (score - score.mean()) / std_score
    )

    data: dict = {
        "Solution 1": rows_idx,
        "Solution 2": cols_idx,
        "wang": wang[rows_idx, cols_idx],
        "jaccard": jacc[rows_idx, cols_idx],
        "discrepancy": disc_vals,
        "score": score,
        "z_score": z_score,
    }
    if labels is not None:
        label_list = list(labels)
        data["label 1"] = [label_list[i] for i in rows_idx]
        data["label 2"] = [label_list[j] for j in cols_idx]

    df = pd.DataFrame(data)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Outlier selection
# ─────────────────────────────────────────────────────────────────────────────


def identify_discrepant_solution_pairs(
    wang_matrix: np.ndarray,
    jaccard_matrix: np.ndarray,
    labels: Sequence[str] | None = None,
    options: DiscrepancyOptions = DiscrepancyOptions(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns both the full pairwise profile and the flagged-outlier subset
    of solution PAIRS.

    Parameters
    ----------
    wang_matrix, jaccard_matrix, labels, options :
        See ``compute_pairwise_discrepancy_profile``.

    Returns
    -------
    pairs_df : pd.DataFrame
        All solution pairs, sorted by score descending.
    outliers_df : pd.DataFrame
        Flagged subset (top-K or z-score ≥ threshold).
    """
    pairs_df = compute_pairwise_discrepancy_profile(
        wang_matrix, jaccard_matrix, labels, options
    )

    if options.top_k is not None:
        if options.top_k <= 0:
            raise ValueError("options.top_k must be a positive integer.")
        outliers_df = pairs_df.head(options.top_k).reset_index(drop=True)
    else:
        outliers_df = pairs_df[pairs_df["z_score"] >= options.z_threshold].reset_index(
            drop=True
        )

    return pairs_df, outliers_df


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────


def plot_discrepancy_summary(
    wang_matrix: np.ndarray,
    jaccard_matrix: np.ndarray,
    outliers_df: pd.DataFrame,
    title: str = "Discrepancia semántica (Wang) vs estructural (Jaccard)",
    reorder: bool = True,
) -> go.Figure:
    """
    Two-panel interactive figure.

    Panel 1 — Scatter Wang vs Jaccard (one point per solution pair)
        Each off-diagonal pair is plotted; colour = discrepancy value.
        Flagged (outlier) pairs are drawn larger with a red outline.
        Diagonal pairs are excluded. Reveals the global relationship
        between the two metrics and where the two disagree most.

    Panel 2 — Heatmap of the full discrepancy matrix
        Rows/cols reordered via hierarchical clustering (when reorder=True
        and n >= 3) so that solutions with similar discrepancy profiles
        appear adjacent, revealing block structure. Tick labels keep the
        original solution index regardless of the new row/column order.

    Parameters
    ----------
    wang_matrix, jaccard_matrix : np.ndarray (n × n)
    outliers_df : pd.DataFrame
        Output of ``identify_discrepant_solution_pairs`` (flagged subset).
    title : str
        Overall figure title.
    reorder : bool
        If True, reorder the heatmap via hierarchical clustering on the
        discrepancy matrix. If False, use the natural solution order.

    Returns
    -------
    go.Figure
    """
    disc = compute_discrepancy_matrix(wang_matrix, jaccard_matrix)
    n = disc.shape[0]
    flagged_pairs = (
        set(zip(outliers_df["Solution 1"], outliers_df["Solution 2"]))
        if not outliers_df.empty
        else set()
    )

    # ── off-diagonal pairs for scatter ──────────────────────────────────────
    rows_idx, cols_idx = np.triu_indices(n, k=1)
    wang_vals = np.asarray(wang_matrix)[rows_idx, cols_idx]
    jacc_vals = np.asarray(jaccard_matrix)[rows_idx, cols_idx]
    disc_vals = disc[rows_idx, cols_idx]
    pair_labels = [f"Sol {r} vs Sol {c}" for r, c in zip(rows_idx, cols_idx)]
    is_flagged = [(r, c) in flagged_pairs for r, c in zip(rows_idx, cols_idx)]
    marker_line_colors = ["#e34948" if f else "rgba(0,0,0,0)" for f in is_flagged]
    marker_sizes = [9 if f else 5 for f in is_flagged]

    # ── heatmap: orden jerárquico o natural ─────────────────────────────────
    order = _hierarchical_order(disc) if reorder and n >= 3 else np.arange(n)

    disc_ordered = disc[np.ix_(order, order)]
    tick_labels = [str(i) for i in order]  # mantiene el índice real como etiqueta

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Wang vs Jaccard por par de soluciones (rojo = {len(flagged_pairs)} outliers)",
            "Heatmap de discrepancia" + (" (ordenado)" if reorder and n >= 3 else ""),
        ),
        column_widths=[0.5, 0.5],
    )

    # Panel 1 · Scatter
    fig.add_trace(
        go.Scatter(
            x=jacc_vals,
            y=wang_vals,
            mode="markers",
            marker={
                "color": disc_vals,
                "colorscale": "RdBu",
                "cmid": 0,
                "size": marker_sizes,
                "opacity": 0.75,
                "line": {"color": marker_line_colors, "width": 1.5},
                "colorbar": {
                    "title": "Wang − Jaccard",
                    "x": 0.46,
                    "len": 0.8,
                },
                "showscale": True,
            },
            text=pair_labels,
            hovertemplate=(
                "%{text}<br>"
                "Jaccard: %{x:.3f}<br>"
                "Wang: %{y:.3f}<br>"
                "Discrepancy: %{marker.color:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    # Diagonal reference line (Wang = Jaccard)
    lim = [0, 1]
    fig.add_trace(
        go.Scatter(
            x=lim,
            y=lim,
            mode="lines",
            line={"color": "gray", "dash": "dash", "width": 1},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Panel 2 · Heatmap (reordenado)
    fig.add_trace(
        go.Heatmap(
            z=disc_ordered,
            x=tick_labels,
            y=tick_labels,
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Wang − Jaccard", "x": 1.01, "len": 0.8},
            hovertemplate=(
                "Sol %{y} vs Sol %{x}<br>" "Discrepancy: %{z:.3f}<extra></extra>"
            ),
            showscale=True,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Similitud Jaccard", row=1, col=1)
    fig.update_yaxes(title_text="Similitud Wang (GO)", row=1, col=1)
    fig.update_xaxes(
        title_text="Solución",
        tickfont_size=8,
        categoryorder="array",
        categoryarray=tick_labels,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="Solución",
        tickfont_size=8,
        categoryorder="array",
        categoryarray=tick_labels,
        row=1,
        col=2,
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=520,
    )
    return fig
