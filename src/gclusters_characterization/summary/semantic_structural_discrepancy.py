"""
semantic_structural_discrepancy.py

Identify clustering solutions that present higher GO semantic similarity
(Wang index) than structural similarity (Jaccard index) to other solutions
— or vice versa.

The core idea is that two solutions can agree biologically (shared GO terms
across their clusters) while assigning genes differently, or assign genes
identically while being biologically divergent. Both cases are detected via
a discrepancy score:

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

compute_solution_discrepancy_profile
    Per-solution summary: mean and max discrepancy against all other
    solutions, plus z-scores for ranking.

identify_discrepant_solutions
    Filter solutions by z-score threshold or fixed top-K, in either
    direction (wang_over_jaccard or jaccard_over_wang).

plot_discrepancy_summary
    Three-panel interactive figure:
      · Scatter Wang vs Jaccard (per pair, coloured by discrepancy)
      · Ranked bar of per-solution mean discrepancy
      · Heatmap of the discrepancy matrix
"""

# ─────────────────────────────────────────────────────────────────────────────
# Libraries
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DiscrepancyOptions:
    """
    Configuration for discrepancy-based outlier detection.

    Attributes
    ----------
    direction : "wang_over_jaccard" | "jaccard_over_wang"
        Which kind of discrepancy to flag.
        · "wang_over_jaccard"  →  Wang HIGH, Jaccard LOW  (default)
        · "jaccard_over_wang"  →  Jaccard HIGH, Wang LOW
    z_threshold : float
        Flag a solution when its mean-discrepancy z-score is ≥ this value.
        Ignored when top_k is set. Default = 1.5.
    top_k : int, optional
        If given, flag exactly the top_k most discrepant solutions instead
        of using the z-score cutoff.
    aggregation : "mean" | "max"
        How to collapse each solution's discrepancy vector (one value per
        pair) into a single per-solution score. "mean" is more robust;
        "max" highlights solutions with at least one very discrepant pair.
    """
    direction: Literal["wang_over_jaccard", "jaccard_over_wang"] = "wang_over_jaccard"
    z_threshold: float = 1.5
    top_k: Optional[int] = None
    aggregation: Literal["mean", "max"] = "mean"


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
        raise ValueError(
            f"Shape mismatch: wang {wang.shape} vs jaccard {jacc.shape}."
        )
    if wang.ndim != 2 or wang.shape[0] != wang.shape[1]:
        raise ValueError("Both matrices must be square (n × n).")

    disc = wang - jacc
    np.fill_diagonal(disc, np.nan)
    return disc


# ─────────────────────────────────────────────────────────────────────────────
# Per-solution profile
# ─────────────────────────────────────────────────────────────────────────────

def compute_solution_discrepancy_profile(
    wang_matrix: np.ndarray,
    jaccard_matrix: np.ndarray,
    solution_matrix: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
    options: DiscrepancyOptions = DiscrepancyOptions(),
) -> pd.DataFrame:
    """
    Per-solution summary of Wang-vs-Jaccard discrepancy.

    For each solution i, the discrepancy vector is disc[i, j] for all j ≠ i.
    A signed score is derived depending on the chosen direction:
        · wang_over_jaccard  →  score = mean(disc[i, :])   (positive = Wang > Jaccard)
        · jaccard_over_wang  →  score = mean(-disc[i, :])  (positive = Jaccard > Wang)

    Parameters
    ----------
    wang_matrix, jaccard_matrix : np.ndarray (n × n)
        See ``compute_discrepancy_matrix``.
    solution_matrix : np.ndarray (n × genes), optional
        Original clustering partitions. If given, adds an "n_clusters" column.
    labels : Sequence[str], optional
        Human-readable solution labels.
    options : DiscrepancyOptions
        Controls direction and aggregation method.

    Returns
    -------
    pd.DataFrame
        Columns: "Solution", ["label"], "score", "mean_wang",
        "mean_jaccard", "z_score", ["n_clusters"].
        Sorted descending by "score".
    """
    disc = compute_discrepancy_matrix(wang_matrix, jaccard_matrix)
    wang = np.asarray(wang_matrix, dtype=np.float64)
    jacc = np.asarray(jaccard_matrix, dtype=np.float64)
    n = disc.shape[0]

    # Raw discrepancy per solution (ignoring diagonal NaN)
    if options.aggregation == "mean":
        raw_score = np.nanmean(disc, axis=1)          # shape (n,)
    else:
        raw_score = np.nanmax(np.abs(disc), axis=1)   # shape (n,)

    # Sign flip for jaccard_over_wang direction
    score = raw_score if options.direction == "wang_over_jaccard" else -raw_score

    std_score = score.std()
    z_score = (
        np.zeros(n) if std_score == 0
        else (score - score.mean()) / std_score
    )

    np.fill_diagonal(wang, np.nan)
    np.fill_diagonal(jacc, np.nan)

    data: dict = {
        "Solution":     np.arange(n),
        "score":        score,
        "mean_wang":    np.nanmean(wang, axis=1),
        "mean_jaccard": np.nanmean(jacc, axis=1),
        "z_score":      z_score,
    }
    if labels is not None:
        if len(labels) != n:
            raise ValueError(f"labels length ({len(labels)}) ≠ n solutions ({n}).")
        data["label"] = list(labels)
    if solution_matrix is not None:
        if len(solution_matrix) != n:
            raise ValueError(
                f"solution_matrix length ({len(solution_matrix)}) ≠ n solutions ({n})."
            )
        data["n_clusters"] = [len(np.unique(row)) for row in solution_matrix]

    df = pd.DataFrame(data)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Outlier selection
# ─────────────────────────────────────────────────────────────────────────────

def identify_discrepant_solutions(
    wang_matrix: np.ndarray,
    jaccard_matrix: np.ndarray,
    solution_matrix: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
    options: DiscrepancyOptions = DiscrepancyOptions(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns both the full profile and the flagged-outlier subset.

    Parameters
    ----------
    wang_matrix, jaccard_matrix, solution_matrix, labels, options :
        See ``compute_solution_discrepancy_profile``.

    Returns
    -------
    profile_df : pd.DataFrame
        All solutions, sorted by score descending.
    outliers_df : pd.DataFrame
        Flagged subset (top-K or z-score ≥ threshold).
    """
    profile_df = compute_solution_discrepancy_profile(
        wang_matrix, jaccard_matrix, solution_matrix, labels, options
    )

    if options.top_k is not None:
        if options.top_k <= 0:
            raise ValueError("options.top_k must be a positive integer.")
        outliers_df = profile_df.head(options.top_k).reset_index(drop=True)
    else:
        outliers_df = (
            profile_df[profile_df["z_score"] >= options.z_threshold]
            .reset_index(drop=True)
        )

    return profile_df, outliers_df


def get_discrepant_solution_indices(outliers_df: pd.DataFrame) -> List[int]:
    """Sorted list of solution indices from ``identify_discrepant_solutions``."""
    if outliers_df.empty:
        return []
    return sorted(int(i) for i in outliers_df["Solution"])


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_discrepancy_summary(
    wang_matrix: np.ndarray,
    jaccard_matrix: np.ndarray,
    profile_df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    title: str = "Discrepancia semántica (Wang) vs estructural (Jaccard)",
) -> go.Figure:
    """
    Three-panel interactive figure.

    Panel 1 — Scatter Wang vs Jaccard (one point per solution pair)
        Each off-diagonal pair is plotted; colour = discrepancy value.
        Diagonal pairs are excluded. Reveals the global relationship
        between the two metrics and where the two disagree most.

    Panel 2 — Ranked bar of per-solution mean discrepancy score
        Outlier solutions highlighted in red.

    Panel 3 — Heatmap of the full discrepancy matrix
        Rows/cols sorted by profile_df order (highest discrepancy first).

    Parameters
    ----------
    wang_matrix, jaccard_matrix : np.ndarray (n × n)
    profile_df : pd.DataFrame
        Output of ``compute_solution_discrepancy_profile``.
    outliers_df : pd.DataFrame
        Output of ``identify_discrepant_solutions`` (flagged subset).
    title : str
        Overall figure title.

    Returns
    -------
    go.Figure
    """
    disc = compute_discrepancy_matrix(wang_matrix, jaccard_matrix)
    n = disc.shape[0]
    flagged = set(outliers_df["Solution"]) if not outliers_df.empty else set()

    # ── off-diagonal pairs for scatter ──────────────────────────────────────
    rows_idx, cols_idx = np.triu_indices(n, k=1)
    wang_vals  = np.asarray(wang_matrix)[rows_idx, cols_idx]
    jacc_vals  = np.asarray(jaccard_matrix)[rows_idx, cols_idx]
    disc_vals  = disc[rows_idx, cols_idx]
    pair_labels = [f"Sol {r} vs Sol {c}" for r, c in zip(rows_idx, cols_idx)]

    # ── per-solution bar ─────────────────────────────────────────────────────
    ranked = profile_df.sort_values("score", ascending=False).reset_index(drop=True)
    bar_colors = [
        "#e34948" if s in flagged else "#2a78d6"
        for s in ranked["Solution"]
    ]
    mean_score = profile_df["score"].mean()
    cutoff = outliers_df["score"].min() if not outliers_df.empty else None

    # ── heatmap order (highest discrepancy first) ────────────────────────────
    order = profile_df["Solution"].tolist()
    disc_ordered = disc[np.ix_(order, order)]
    tick_labels = [str(s) for s in order]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Wang vs Jaccard por par de soluciones",
            f"Score por solución (rojo = {len(flagged)} outliers)",
            "Heatmap de discrepancia",
        ),
        column_widths=[0.35, 0.35, 0.30],
    )

    # Panel 1 · Scatter
    fig.add_trace(
        go.Scatter(
            x=jacc_vals,
            y=wang_vals,
            mode="markers",
            marker=dict(
                color=disc_vals,
                colorscale="RdBu",
                cmid=0,
                size=5,
                opacity=0.6,
                colorbar=dict(
                    title="Wang − Jaccard",
                    x=0.62,
                    len=0.8,
                ),
                showscale=True,
            ),
            text=pair_labels,
            hovertemplate=(
                "%{text}<br>"
                "Jaccard: %{x:.3f}<br>"
                "Wang: %{y:.3f}<br>"
                "Discrepancy: %{marker.color:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1, col=1,
    )
    # Diagonal reference line (Wang = Jaccard)
    lim = [0, 1]
    fig.add_trace(
        go.Scatter(
            x=lim, y=lim,
            mode="lines",
            line=dict(color="gray", dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # Panel 2 · Ranked bar
    fig.add_trace(
        go.Bar(
            x=list(range(len(ranked))),
            y=ranked["score"],
            marker_color=bar_colors,
            customdata=ranked["Solution"],
            hovertemplate=(
                "Solution: %{customdata}<br>"
                "Score: %{y:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_hline(
        y=mean_score, line_dash="dash", line_color="black",
        annotation_text=f"Media: {mean_score:.2f}",
        annotation_position="top right",
        row=1, col=2,
    )
    if cutoff is not None:
        fig.add_hline(
            y=cutoff, line_color="#e34948", line_width=2,
            annotation_text=f"Umbral: {cutoff:.2f}",
            annotation_position="bottom right",
            row=1, col=2,
        )

    # Panel 3 · Heatmap
    fig.add_trace(
        go.Heatmap(
            z=disc_ordered,
            x=tick_labels,
            y=tick_labels,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Wang − Jaccard", x=1.01, len=0.8),
            hovertemplate=(
                "Sol %{y} vs Sol %{x}<br>"
                "Discrepancy: %{z:.3f}<extra></extra>"
            ),
            showscale=True,
        ),
        row=1, col=3,
    )

    fig.update_xaxes(title_text="Similitud Jaccard", row=1, col=1)
    fig.update_yaxes(title_text="Similitud Wang (GO)", row=1, col=1)
    fig.update_xaxes(title_text="Soluciones (ordenadas por score)", row=1, col=2)
    fig.update_yaxes(title_text="Score (Wang − Jaccard)", row=1, col=2)
    fig.update_xaxes(title_text="Solución", tickfont_size=8, row=1, col=3)
    fig.update_yaxes(title_text="Solución", tickfont_size=8, row=1, col=3)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=520,
        bargap=0.04,
    )
    return fig