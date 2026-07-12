"""
consensus_distance_summary: Identify clustering solutions that are significantly distant from a single,
designated consensus solution (e.g. ``consensus_arrays[best_idx]``), based on a partition-to-partition 
distance measure computed elsewhere (typically 1 - Jaccard between gene co-assignments, via
``jaccard_values.jaccard_index_solutions``).

Functions
1. compute_consensus_distance_scores       - Per-solution distance-to-consensus, plus its z-score relative to the 
                                            distribution of distances across all solutions.
2. identify_outlier_solutions_vs_consensus - Filter to only the solutions flagged as significantly distant from the
                                            consensus solution — either via z-score threshold, or a fixed top-K.
3. get_outlier_solution_indices            - Unique, sorted list of flagged solution indices.
4. plot_consensus_distance_summary         - Two-panel interactive figure: ranked bar chart of distances (outliers
                                            highlighted) + histogram of the distance distribution with the cutoff.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses     import dataclass
from typing          import List, Optional, Sequence
from plotly.subplots import make_subplots
import numpy                as np
import pandas               as pd
import plotly.graph_objects as go


# ──────────────────────────────────────────────────────────────────────────────
# Options
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConsensusDistanceOptions:
    """
    Configuration for outlier-vs-consensus detection.

    Attributes
    ----------
    z_threshold : float
        A solution is flagged when its z-score (of distance to the
        consensus solution) is at or above this value. Positive by
        convention, since we're looking for solutions MORE distant than
        typical. 1.5-2.0 is a reasonable starting range. Ignored if
        ``top_k`` is set.
    top_k : int, optional
        If given, flags exactly the ``top_k`` most distant solutions,
        overriding ``z_threshold`` — matches a fixed "examine the top N
        outliers" workflow instead of a statistical cutoff.
    """
    z_threshold: float   = 1.5
    top_k: Optional[int] = None


# ──────────────────────────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────────────────────────

def compute_consensus_distance_scores(
    distances_to_consensus: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    solution_matrix: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Build the per-solution distance-to-consensus table, with a z-score
    relative to the distribution of all distances.

    Parameters
    ----------
    distances_to_consensus : Sequence[float]
        Distance of every solution to the single reference consensus
        solution (e.g. ``1 - jaccard_index_solutions(...)`` computed
        pairwise against ``consensus_solution``, one value per solution).
    labels : Sequence[str], optional
        Human-readable solution labels. If given, added as a "label" column.
    solution_matrix : np.ndarray, optional
        The original solutions (one partition per row). If given, adds an
        "n_clusters" column (number of unique labels per solution) for
        context, matching the original ranking table.

    Returns
    -------
    pd.DataFrame
        Columns: "Solution", ["label"], "distance", "z_score",
        ["n_clusters"]. Sorted descending by "distance" (most distant
        solution first).
    """
    distances = np.asarray(distances_to_consensus, dtype=np.float64)
    n = len(distances)

    if n < 3:
        raise ValueError("distances_to_consensus must contain at least 3 solutions.")
    if labels is not None and len(labels) != n:
        raise ValueError(f"labels length ({len(labels)}) must match n solutions ({n}).")
    if solution_matrix is not None and len(solution_matrix) != n:
        raise ValueError(f"solution_matrix length ({len(solution_matrix)}) must match n solutions ({n}).")

    std_dist = distances.std()
    z_score = np.zeros_like(distances) if std_dist == 0 else (distances - distances.mean()) / std_dist

    data = {
        "Solution": np.arange(n),
        "distance": distances,
        "z_score": z_score,
    }
    if labels is not None:
        data["label"] = list(labels)
    if solution_matrix is not None:
        data["n_clusters"] = [len(np.unique(row)) for row in solution_matrix]

    df = pd.DataFrame(data)
    return df.sort_values("distance", ascending=False).reset_index(drop=True)


def identify_outlier_solutions_vs_consensus(
    distances_to_consensus: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    solution_matrix: Optional[np.ndarray] = None,
    options: ConsensusDistanceOptions = ConsensusDistanceOptions(),
) -> pd.DataFrame:
    """
    Solutions significantly distant from the reference consensus solution —
    either the top-K most distant (if ``options.top_k`` is set) or those
    whose z-score is at or above ``options.z_threshold``.

    Parameters
    ----------
    distances_to_consensus, labels, solution_matrix :
        See ``compute_consensus_distance_scores``.
    options : ConsensusDistanceOptions
        Selection criterion (z-score threshold or fixed top-K).

    Returns
    -------
    pd.DataFrame
        Subset of ``compute_consensus_distance_scores``'s output, sorted by
        distance descending (most distant first).
    """
    scores_df = compute_consensus_distance_scores(distances_to_consensus, labels, solution_matrix)

    if options.top_k is not None:
        if options.top_k <= 0:
            raise ValueError("options.top_k must be a positive integer.")
        return scores_df.head(options.top_k).reset_index(drop=True)

    return scores_df[scores_df["z_score"] >= options.z_threshold].reset_index(drop=True)


def get_outlier_solution_indices(outliers_df: pd.DataFrame) -> List[int]:
    """
    Sorted list of solution indices from
    ``identify_outlier_solutions_vs_consensus``'s output.

    Parameters
    ----------
    outliers_df : pd.DataFrame
        Must contain a "Solution" column.

    Returns
    -------
    List[int]
    """
    if outliers_df.empty:
        return []
    return sorted(int(i) for i in outliers_df["Solution"])


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_consensus_distance_summary(
    scores_df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    title: str = "Soluciones outlier respecto a la solución de consenso",
) -> go.Figure:
    """
    Two-panel interactive figure:
      - Left: every solution's distance to consensus, ranked descending,
        with the flagged outliers highlighted in red and the mean marked.
      - Right: histogram of the distance distribution, with the mean and
        the cutoff (minimum distance among the flagged solutions) marked.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Output of ``compute_consensus_distance_scores`` (all solutions).
    outliers_df : pd.DataFrame
        Output of ``identify_outlier_solutions_vs_consensus`` (flagged
        subset) — used to highlight bars and determine the cutoff line.
    title : str
        Overall figure title.

    Returns
    -------
    go.Figure
    """
    if scores_df.empty:
        return go.Figure()

    ranked = scores_df.sort_values("distance", ascending=False).reset_index(drop=True)
    flagged_solutions = set(outliers_df["Solution"]) if not outliers_df.empty else set()
    is_flagged = ranked["Solution"].isin(flagged_solutions)

    mean_dist = scores_df["distance"].mean()
    cutoff = outliers_df["distance"].min() if not outliers_df.empty else None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Ranking de distancia al consenso (rojo = {len(flagged_solutions)} outliers)",
            "Distribución de distancias al consenso",
        ),
    )

    # ── Panel 1: ranked bar chart ────────────────────────────────────────────
    bar_colors = np.where(is_flagged, "#e34948", "#2a78d6")
    fig.add_trace(
        go.Bar(
            x=list(range(len(ranked))),
            y=ranked["distance"],
            marker_color=bar_colors,
            customdata=ranked["Solution"],
            hovertemplate="Solution: %{customdata}<br>Distance: %{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_hline(
        y=mean_dist, line_dash="dash", line_color="black",
        annotation_text=f"Media: {mean_dist:.2f}", annotation_position="top right",
        row=1, col=1,
    )

    # ── Panel 2: histogram ───────────────────────────────────────────────────
    fig.add_trace(
        go.Histogram(
            x=scores_df["distance"],
            nbinsx=30,
            marker_color="#2a78d6",
            opacity=0.75,
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_vline(
        x=mean_dist, line_dash="dash", line_color="black",
        annotation_text=f"Media: {mean_dist:.2f}", annotation_position="top",
        row=1, col=2,
    )
    if cutoff is not None:
        fig.add_vline(
            x=cutoff, line_color="#e34948", line_width=2,
            annotation_text=f"Umbral: {cutoff:.2f}", annotation_position="top left",
            row=1, col=2,
        )

    fig.update_xaxes(title_text="Soluciones (ordenadas por distancia)", row=1, col=1)
    fig.update_yaxes(title_text="Distancia al consenso (1 − Jaccard)", row=1, col=1)
    fig.update_xaxes(title_text="Distancia (1 − Jaccard)", row=1, col=2)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=2)

    fig.update_layout(title=title, template="plotly_white", bargap=0.05)

    return fig