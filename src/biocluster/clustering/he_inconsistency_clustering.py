"""
he_inconsistency_clustering: This module provides utilities for automatic (inconsistency
-coefficient-based) hierarchical clustering using a distance matrix representation of
elements (e.g., genes).

The module supports:
- Automatic detection of the number of clusters via the inconsistency coefficient of the linkage matrix.
- Optional creation of interactive two-panel visualizations (dendrogram + inconsistency profile) using Plotly.
- Export of figures to HTML format.

Functions:
1. compute_inconsistency_clustering – Automatically detect the number of clusters using the inconsistency coefficient of the linkage matrix.
2. he_inconsistency_clustering      – High-level interface for automatic (inconsistency-based) clustering, two-panel visualization, and export.
3. _validate_distance_matrix        – Validate structural properties of a distance matrix.
4. _validate_genes                  – Validate gene labels associated with the distance matrix.
5. _build_inconsistency_figure      – Construct a two-panel figure (dendrogram + inconsistency profile).
6. _figure_to_html                  – Convert a Plotly figure into HTML.
7. _write_text                      – Write HTML output to disk.
8. _detect_inconsistency_cut        – Detect the optimal cut point via the inconsistency coefficient.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import (
    cophenet,
    dendrogram,
    fcluster,
    inconsistent,
    linkage,
)
from scipy.spatial.distance import squareform

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data Types
# ──────────────────────────────────────────────────────────────────────────────
PathLike = Union[str, Path]
LinkageMethod = Literal[
    "single", "complete", "average", "weighted", "centroid", "median", "ward"
]


# ──────────────────────────────────────────────────────────────────────────────
# Classes
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class InconsistencyClusteringOptions:
    """
    Options for automatic (inconsistency-coefficient-based) hierarchical
    clustering.

    Instead of requiring the user to specify ``num_groups``, this approach
    examines every merge in the linkage matrix ``Z`` and computes its
    *inconsistency coefficient* (scipy.cluster.hierarchy.inconsistent):
    for each fusion, the height is compared against the mean and standard
    deviation of the fusion heights found in its local neighbourhood
    (governed by ``depth``), rather than against the dendrogram as a whole.
    A merge whose height is unusually large *relative to its own
    neighbourhood* marks a natural cluster boundary. Cutting just below
    the most anomalous merge yields the most natural number of clusters.

    This is more robust than a raw "largest height jump" rule when
    different branches of the tree operate at different distance scales
    (common with real similarity/distance data), because the coefficient
    is locally normalized instead of being an absolute height difference.

    Attributes:
        method : LinkageMethod
            Linkage algorithm passed to ``scipy.cluster.hierarchy.linkage``.
        depth : int
            Number of levels below each link to include when computing the
            local mean/std of merge heights. Passed directly to
            ``scipy.cluster.hierarchy.inconsistent``. Default 2 (scipy's
            own default). Larger values smooth the local estimate over a
            wider neighbourhood; smaller values make it more sensitive to
            very local structure.
        n_candidates : int
            Number of top candidate cut points to consider, ranked by
            inconsistency coefficient (descending). When ``n_candidates=1``
            (default) only the single most anomalous merge is used.
            Increasing it lets you inspect a ranked list of alternatives
            via the ``inconsistency_report`` returned by
            ``compute_inconsistency_clustering``.
        min_clusters : int
            Lower bound on the number of clusters accepted. Any candidate
            that would produce fewer clusters than this is skipped. Default 2.
        max_clusters : int or None
            Upper bound on the number of clusters. Any candidate that would
            produce more clusters than this is skipped. ``None`` = no limit.
        validate_distance : bool
            Enable strict distance-matrix structural checks.
        sym_tol : float
            Numerical tolerance used in symmetry and diagonal checks.
        verbose : bool
            If ``True``, prints status messages (always logs regardless).
    """

    method: LinkageMethod = "average"
    depth: int = 2
    n_candidates: int = 1
    min_clusters: int = 2
    max_clusters: Optional[int] = None
    validate_distance: bool = True
    sym_tol: float = 1e-12
    verbose: bool = True


@dataclass(frozen=True)
class DendrogramOptions:
    """
    Options to build the dendrogram figure.

    Attributes:
        title: plot title base.
        height: figure height in pixels.
        width: figure width in pixels.
        show_cutoff: show a horizontal cutoff line for num_groups (when > 1).
        tick_angle: x tick label angle.
        line_color: dendrogram line color.
        cutoff_color: cutoff line color.
        cutoff_dash: cutoff line dash style.
    """

    title: str = "Dendrogram"
    height: int = 1080
    width: int = 1920
    show_cutoff: bool = True
    tick_angle: int = 45
    line_color: str = "black"
    cutoff_color: str = "red"
    cutoff_dash: str = "dash"


@dataclass(frozen=True)
class ExportOptions:
    """
    Options for exporting HTML.

    Attributes:
        include_plotlyjs: 'cdn' keeps HTML small; 'embed' is standalone but heavier.
        full_html: if True writes a full HTML document; else a div snippet.
        verbose: if True prints status messages (still logs always).
    """

    include_plotlyjs: Union[Literal["cdn", "embed"], bool] = "cdn"
    full_html: bool = False
    verbose: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Internal Functions
# ──────────────────────────────────────────────────────────────────────────────
def _as_path(p: PathLike) -> Path:
    """Ensure the input is converted to a Path object."""
    return p if isinstance(p, Path) else Path(p)


def _log_or_print(msg: str, verbose: bool) -> None:
    """
    Convert an input path-like object to a Path instance.

    Parameters
    ----------
    p : PathLike
        Input path provided as string or Path.

    Returns
    -------
    Path
        Normalized Path object.
    """
    logger.info(msg)
    if verbose:
        print(msg)


def _validate_genes(genes: Sequence[str], n: int) -> list[str]:
    """
    Validate gene identifiers against the expected matrix size.

    Parameters
    ----------
    genes : Sequence[str]
        Gene identifiers corresponding to the distance matrix rows.
    n : int
        Expected number of genes.

    Returns
    -------
    List[str]
        List of validated gene names converted to strings.

    Raises
    ------
    ValueError
        If the number of genes does not match the matrix size.
    """
    if not isinstance(genes, (list, tuple)):
        genes = list(genes)

    if len(genes) != n:
        raise ValueError(
            f"Number of genes ({len(genes)}) does not match matrix size ({n})."
        )

    # Ensure all are strings (best-effort)
    out = [str(g) for g in genes]
    return out


def _validate_distance_matrix(distance_matrix: np.ndarray, tol: float) -> None:
    """
    Validate that the provided matrix is a proper distance matrix.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Square matrix representing pairwise distances.
    tol : float
        Numerical tolerance used for symmetry and diagonal checks.

    Raises
    ------
    TypeError
        If matrix is not numeric or not a numpy array.
    ValueError
        If matrix is not square, symmetric, contains NaN values,
        negative values, or a non-zero diagonal.
    """
    if not isinstance(distance_matrix, np.ndarray):
        raise TypeError(
            f"distance_matrix must be a numpy.ndarray, got: {type(distance_matrix)}"
        )

    if distance_matrix.ndim != 2:
        raise ValueError(f"distance_matrix must be 2D, got ndim={distance_matrix.ndim}")

    n, m = distance_matrix.shape
    if n != m:
        raise ValueError("Distance matrix must be square.")

    if n < 2:
        raise ValueError("Distance matrix must be at least 2x2.")

    if not np.issubdtype(distance_matrix.dtype, np.number):
        raise TypeError(
            f"distance_matrix must be numeric dtype, got: {distance_matrix.dtype}"
        )

    if np.isnan(distance_matrix).any():
        raise ValueError("Distance matrix contains NaN values.")

    if np.any(distance_matrix < 0):
        raise ValueError("Distance matrix cannot contain negative values.")

    # Symmetry check (important for squareform)
    if not np.allclose(distance_matrix, distance_matrix.T, atol=tol, rtol=0):
        raise ValueError("Distance matrix must be symmetric (within tolerance).")

    # Diagonal should be ~0 for a distance matrix
    diag = np.diag(distance_matrix)
    if not np.allclose(diag, 0.0, atol=tol, rtol=0):
        raise ValueError("Distance matrix diagonal must be 0 (within tolerance).")


def _build_inconsistency_figure(
    Z: np.ndarray,
    genes: Sequence[str],
    R: np.ndarray,
    candidates: list[dict],
    opts: DendrogramOptions,
) -> go.Figure:
    """
    Build a two-panel interactive figure: dendrogram (left) + inconsistency
    coefficient profile (right), sharing the distance/height axis.

    The right panel plots, for every merge in ``Z``, its inconsistency
    coefficient against the height at which that merge occurred. Every
    candidate cut in ``candidates`` (i.e. as many as the user requested via
    ``InconsistencyClusteringOptions.n_candidates``) is drawn as a labelled
    horizontal reference line across both panels, plus a highlighted marker
    at its exact flagged merge in the right panel -- so the user can
    visually compare all the alternatives, not just the top-ranked one.
    The dendrogram's branch coloring always follows the top-ranked (rank 1)
    candidate.

    Parameters
    ----------
    Z : numpy.ndarray
        Linkage matrix (n-1 x 4).
    genes : Sequence[str]
        Gene identifiers corresponding to matrix rows/columns.
    R : numpy.ndarray
        Inconsistency matrix (n-1 x 4) as returned by
        ``scipy.cluster.hierarchy.inconsistent`` (column 3 = coefficient).
    candidates : List[dict]
        The ``inconsistency_report`` produced by
        ``compute_inconsistency_clustering`` -- one entry per requested
        candidate (``rank``, ``k``, ``coefficient``, ``cut_height``,
        ``merge_height``, ``labels``, ``selected``). Its length equals
        ``n_candidates`` as set by the user.
    opts : DendrogramOptions
        Visual configuration shared with the single-panel dendrogram.

    Returns
    -------
    plotly.graph_objects.Figure
        Two-panel figure ready for display or HTML export.
    """
    if not candidates:
        raise ValueError("candidates must contain at least one entry.")

    gene_labels = [f"{i}-{g}" for i, g in enumerate(genes)]
    best = candidates[0]  # rank 1 -- used to color the dendrogram branches

    _prev_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(_prev_limit, len(Z) * 3 + 1000))
    try:
        ddata = dendrogram(
            Z,
            labels=gene_labels,
            color_threshold=best["cut_height"],
            above_threshold_color=opts.line_color,
            no_plot=True,
        )
    finally:
        sys.setrecursionlimit(_prev_limit)

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.72, 0.28],
        horizontal_spacing=0.03,
        subplot_titles=("Dendrogram", "Inconsistency coefficient"),
    )

    # Get matplotlib default color cycle
    mpl_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Left panel: dendrogram segments
    for x, y, color in zip(ddata["icoord"], ddata["dcoord"], ddata["color_list"]):
        try:
            if isinstance(color, str) and color.startswith("C"):
                idx = int(color[1:])
                color = mpl_cycle[idx % len(mpl_cycle)]
            color = mcolors.to_hex(color)
        except (ValueError, TypeError):
            color = opts.line_color

        xs = [x[0], x[1], None, x[1], x[2], None, x[2], x[3], None]
        ys = [y[0], y[1], None, y[1], y[2], None, y[2], y[3], None]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": color},
                hoverinfo="none",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    leaf_idx = ddata["leaves"]
    leaf_labels = [gene_labels[i] for i in leaf_idx]
    leaf_positions = [5 + 10 * i for i in range(len(leaf_idx))]

    # Right panel: inconsistency coefficient per merge, y = fusion height.
    # Candidate merges are highlighted; everything else stays neutral.
    heights = Z[:, 2]
    coefficients = R[:, 3]
    candidate_merge_heights = {c["merge_height"] for c in candidates}

    point_colors = [
        opts.cutoff_color if h in candidate_merge_heights else opts.line_color
        for h in heights
    ]
    point_sizes = [11 if h in candidate_merge_heights else 6 for h in heights]

    fig.add_trace(
        go.Scatter(
            x=coefficients,
            y=heights,
            mode="markers",
            marker={
                "color": point_colors,
                "size": point_sizes,
                "line": {"width": 0.5, "color": "black"},
            },
            hovertemplate="Altura: %{y:.4f}<br>Coeficiente: %{x:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # One reference line + annotation per requested candidate, ranked by
    # opacity so the best candidate stands out while the rest remain visible
    # for comparison.
    if opts.show_cutoff:
        x_range_left = [0, 10 * len(leaf_idx)]
        max_coeff = float(np.max(coefficients)) if len(coefficients) else 1.0

        for entry in candidates:
            rank = entry["rank"]
            cut_h = entry["cut_height"]
            is_best = rank == 1
            # Rank 1 fully opaque; further ranks fade out progressively.
            opacity = 1.0 if is_best else max(0.35, 1.0 - 0.2 * (rank - 1))
            width = 2.5 if is_best else 1.5
            dash = opts.cutoff_dash if is_best else "dot"

            fig.add_trace(
                go.Scatter(
                    x=x_range_left,
                    y=[cut_h, cut_h],
                    mode="lines",
                    line={"color": opts.cutoff_color, "dash": dash, "width": width},
                    opacity=opacity,
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, max_coeff * 1.15],
                    y=[cut_h, cut_h],
                    mode="lines",
                    line={"color": opts.cutoff_color, "dash": dash, "width": width},
                    opacity=opacity,
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            # Rank/k annotation next to the line in the right panel.
            fig.add_annotation(
                x=max_coeff * 1.17,
                y=cut_h,
                text=f"#{rank}: k={entry['k']}",
                showarrow=False,
                xanchor="left",
                font={"size": 11, "color": opts.cutoff_color},
                opacity=opacity,
                row=1,
                col=2,
            )

    title = opts.title
    if best["k"] > 1:
        n_shown = len(candidates)
        suffix = (
            f"mejor: {best['k']} clusters"
            if n_shown == 1
            else f"mejor: {best['k']} clusters -- {n_shown} candidatos mostrados"
        )
        title = f"{opts.title} ({suffix}, corte por coeficiente de inconsistencia)"

    fig.update_layout(
        title=title,
        height=opts.height,
        width=opts.width,
        hovermode="closest",
        margin={"r": 90},  # room for candidate annotations
    )
    fig.update_xaxes(
        title_text="Genes (Index-Name)",
        tickmode="array",
        tickvals=leaf_positions,
        ticktext=leaf_labels,
        tickangle=opts.tick_angle,
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Coeficiente de inconsistencia", row=1, col=2)
    fig.update_yaxes(title_text="Distancia", row=1, col=1)
    fig.update_yaxes(row=1, col=2, showticklabels=False)

    return fig


def _figure_to_html(fig: go.Figure, export: ExportOptions) -> str:
    """
    Convert a Plotly figure to HTML representation.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure object.
    export : ExportOptions
        Export configuration.

    Returns
    -------
    str
        HTML representation of the figure.
    """
    return fig.to_html(
        include_plotlyjs=export.include_plotlyjs, full_html=export.full_html
    )


def _write_text(filepath: PathLike, content: str) -> Path:
    """
    Write text content to a file.

    Parameters
    ----------
    filepath : PathLike
        Destination path.
    content : str
        Text content to write.

    Returns
    -------
    Path
        Path where the file was written.
    """
    p = _as_path(filepath)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic clustering helpers
# ──────────────────────────────────────────────────────────────────────────────


def _detect_inconsistency_cut(
    Z: np.ndarray,
    depth: int,
    n_candidates: int,
    min_clusters: int,
    max_clusters: Optional[int],
) -> tuple[list[tuple[int, float, float, float]], np.ndarray]:
    """
    Identify candidate cut points in a linkage matrix using the
    inconsistency coefficient.

    For every merge step, ``scipy.cluster.hierarchy.inconsistent`` computes
    the mean and standard deviation of the fusion heights among the links
    found within ``depth`` levels below it, then expresses the merge's own
    height as a number of local standard deviations above that mean — the
    *inconsistency coefficient*. Unlike a raw height jump (which only
    compares two consecutive heights in absolute terms), this coefficient
    adapts to how the density/scale of merges varies across the dendrogram,
    which matters whenever different branches operate at different
    distance scales.

    Parameters
    ----------
    Z : numpy.ndarray
        Linkage matrix (n-1 x 4) from ``scipy.cluster.hierarchy.linkage``.
    depth : int
        Number of levels below each link considered for the local
        mean/std calculation. Passed to ``scipy.cluster.hierarchy.inconsistent``.
    n_candidates : int
        How many top candidates to return (ranked by inconsistency
        coefficient, descending).
    min_clusters : int
        Minimum number of clusters a candidate must yield to be included.
    max_clusters : int or None
        Maximum number of clusters allowed. ``None`` = no upper limit.

    Returns
    -------
    candidates : List[Tuple[int, float, float, float]]
        Each entry is ``(k, coefficient, cut_height, merge_height)`` where:

        - ``k``            — number of clusters obtained by cutting here.
        - ``coefficient``  — inconsistency coefficient of the flagged merge
                              (larger = more anomalous = more "natural" boundary).
        - ``cut_height``   — distance threshold used for ``fcluster``
                              (midpoint just below the flagged merge).
        - ``merge_height``  — actual fusion height of the flagged merge
                              (useful for plotting the exact point that was
                              flagged, as opposed to the midpoint cut).
    R : numpy.ndarray
        Full inconsistency matrix (n-1 x 4) as returned by scipy: columns
        are [local mean height, local std height, link count, inconsistency
        coefficient]. Returned so it can be reused for visualization/
        reporting without recomputation.

    Notes
    -----
    Cutting *just below* the flagged merge (rather than at its own height)
    prevents that anomalous fusion from being carried out, which is what
    actually separates the two groups it would otherwise have joined.
    """
    heights = Z[:, 2]
    n_steps = len(heights)  # == n - 1
    n_leaves = n_steps + 1  # original number of elements

    R = inconsistent(Z, d=depth)
    coefficients = R[:, 3]

    # Build candidate list: (merge_index, k_clusters, coefficient)
    candidates = []
    for i in range(n_steps):
        # Clusters remaining if merges 0..i-1 are performed but merge i is not.
        k = n_leaves - i
        if k < min_clusters:
            continue
        if max_clusters is not None and k > max_clusters:
            continue
        candidates.append((i, k, float(coefficients[i])))

    if not candidates:
        raise ValueError(
            f"No valid cut point found with min_clusters={min_clusters} "
            f"and max_clusters={max_clusters}.  Try relaxing these bounds."
        )

    # Sort by inconsistency coefficient descending, take top n_candidates
    candidates.sort(key=lambda x: x[2], reverse=True)
    top = candidates[:n_candidates]

    result = []
    for idx, k, coeff in top:
        if idx == 0:
            # No previous merge to anchor on; cut just below the first fusion.
            cut_height = float(heights[0] / 2.0) if heights[0] > 0 else 0.0
        else:
            cut_height = float((heights[idx - 1] + heights[idx]) / 2.0)
        result.append((k, coeff, cut_height, float(heights[idx])))

    return result, R


# ──────────────────────────────────────────────────────────────────────────────
# Main Functions
# ──────────────────────────────────────────────────────────────────────────────
def compute_inconsistency_clustering(
    distance_matrix: np.ndarray,
    genes: "Sequence[str]",
    options: "Optional[InconsistencyClusteringOptions]" = None,
) -> "tuple[np.ndarray, np.ndarray, float, list[dict], np.ndarray]":
    """
    Perform hierarchical clustering with automatic cluster-count detection
    based on the inconsistency coefficient.

    Unlike ``compute_hierarchical_clustering``, this function does **not**
    require the user to specify ``num_groups``. Instead it flags the merge
    step whose fusion height is most anomalous *relative to its local
    neighbourhood* in the dendrogram -- the **inconsistency coefficient**
    heuristic (``scipy.cluster.hierarchy.inconsistent``) -- and cuts the
    tree just below it.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Square pairwise distance matrix (n x n), values in [0, inf).
        The diagonal must be zero and the matrix must be symmetric.
    genes : Sequence[str]
        Gene identifiers corresponding to matrix rows/columns.
    options : InconsistencyClusteringOptions
        Configuration for clustering method, inconsistency search depth,
        and cluster-count constraints.

    Returns
    -------
    Z : numpy.ndarray
        Linkage matrix (n-1 x 4) from scipy.cluster.hierarchy.linkage.
    labels : numpy.ndarray of int, shape (n,)
        Cluster label (1-based) assigned to each gene.
    cophenetic_corr : float
        Cophenetic correlation coefficient -- how faithfully the dendrogram
        preserves the original distances. Values > 0.75 indicate a strong
        clustering structure.
    inconsistency_report : List[dict]
        Ranked list of the top-n_candidates candidate cut points, each with:

        - 'rank'         -- 1 = best (largest inconsistency coefficient).
        - 'k'            -- number of clusters at this cut.
        - 'coefficient'  -- inconsistency coefficient of the flagged merge.
        - 'cut_height'   -- distance threshold used for fcluster.
        - 'merge_height' -- actual fusion height of the flagged merge
                             (the exact point highlighted in the figure).
        - 'labels'       -- numpy.ndarray of int, cluster labels (1-based)
                             obtained by cutting at *this* candidate's
                             cut_height -- lets you compare candidates
                             without recomputing fcluster yourself.
        - 'selected'     -- True only for the top-ranked candidate (rank 1).
          The value returned as ``labels`` below always corresponds to
          rank 1; use ``inconsistency_report[i]['labels']`` directly to
          adopt a different candidate.
    R : numpy.ndarray
        Full inconsistency matrix returned by scipy (n-1 x 4), kept around
        so it can be reused directly for visualization (see
        ``he_inconsistency_clustering`` / ``_build_inconsistency_figure``)
        without recomputing it.

    Raises
    ------
    ValueError
        If the distance matrix is invalid, gene list length is wrong, or no
        valid cut point exists within the min/max_clusters constraints.

    Examples
    --------
    >>> import numpy as np
    >>> from he_inconsistency_clustering import compute_inconsistency_clustering, InconsistencyClusteringOptions
    >>>
    >>> dist = np.array([[0,   0.2, 0.9, 0.8],
    ...                  [0.2, 0,   0.85,0.75],
    ...                  [0.9, 0.85,0,   0.15],
    ...                  [0.8, 0.75,0.15,0   ]])
    >>> genes = ["GeneA", "GeneB", "GeneC", "GeneD"]
    >>>
    >>> Z, labels, coph, report, R = compute_inconsistency_clustering(dist, genes)
    >>> print(labels)        # e.g. [1 1 2 2]
    >>> print(report[0])     # best cut details
    """
    if options is None:
        options = InconsistencyClusteringOptions()

    if options.validate_distance:
        _validate_distance_matrix(distance_matrix, tol=options.sym_tol)

    n = distance_matrix.shape[0]
    _validate_genes(genes, n)

    if n < 3:
        raise ValueError("Inconsistency-based clustering requires at least 3 elements.")

    # Build linkage
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method=options.method)

    # Cophenetic correlation
    cophenetic_corr, _ = cophenet(Z, condensed)

    interpretation = (
        "strong"
        if cophenetic_corr > 0.75
        else "moderate" if cophenetic_corr > 0.5 else "weak"
    )
    _log_or_print(
        f"[inconsistency] Cophenetic correlation: {cophenetic_corr:.4f} ({interpretation})",
        options.verbose,
    )

    # Detect inconsistency-based cut points
    candidates, R = _detect_inconsistency_cut(
        Z,
        depth=options.depth,
        n_candidates=options.n_candidates,
        min_clusters=options.min_clusters,
        max_clusters=options.max_clusters,
    )

    # Best cut is the first (highest inconsistency coefficient)
    best_k, best_coeff, best_cut_height, best_merge_height = candidates[0]

    _log_or_print(
        f"[inconsistency] Highest coefficient: {best_coeff:.4f} -- cutting at height "
        f"{best_cut_height:.6f} -> {best_k} clusters.",
        options.verbose,
    )

    # Build inconsistency report, computing labels for *every* candidate so
    # the user can compare/adopt any of them without recomputing fcluster.
    inconsistency_report = []
    for rank, (k, coeff, cut_height, merge_height) in enumerate(candidates, start=1):
        candidate_labels = fcluster(Z, t=cut_height, criterion="distance")
        inconsistency_report.append(
            {
                "rank": rank,
                "k": k,
                "coefficient": coeff,
                "cut_height": cut_height,
                "merge_height": merge_height,
                "labels": candidate_labels,
                "selected": rank == 1,
            }
        )

    # Primary labels returned correspond to the top-ranked (rank 1) candidate.
    labels = inconsistency_report[0]["labels"]

    if options.verbose and len(inconsistency_report) > 1:
        _log_or_print(
            "[inconsistency] Alternative cut points (ranked by coefficient):",
            options.verbose,
        )
        for entry in inconsistency_report[1:]:
            _log_or_print(
                f"  rank={entry['rank']}  k={entry['k']}  "
                f"coeff={entry['coefficient']:.4f}  height={entry['cut_height']:.6f}",
                options.verbose,
            )

    return Z, labels, cophenetic_corr, inconsistency_report, R


def he_inconsistency_clustering(
    distance_matrix: np.ndarray,
    genes: Sequence[str],
    clustering: InconsistencyClusteringOptions = InconsistencyClusteringOptions(),
    dendrogram_opts: DendrogramOptions = DendrogramOptions(),
    save_html_to: Optional[PathLike] = None,
    export: ExportOptions = ExportOptions(),
    selected_rank: int = 1,
    return_fig: bool = False,
    return_html: bool = False,
    return_report: bool = False,
):
    """
    High-level function for automatic, inconsistency-coefficient-based
    clustering:
    - Computes hierarchical clustering with the number of groups detected
      automatically (no ``num_groups`` required).
    - Evaluates as many candidate cut points as requested via
      ``clustering.n_candidates`` -- not just the single best one.
    - Optionally builds a two-panel interactive figure (dendrogram +
      inconsistency profile) via Plotly, showing *every* requested
      candidate as a labelled reference line, so the user can visually
      compare alternatives before committing to one.
    - Optionally exports the figure to HTML.

    Args:
        distance_matrix: square distance matrix (n x n).
        genes: gene identifiers (len n).
        clustering: inconsistency-clustering configuration (method, depth,
            n_candidates -- controls how many cut candidates are computed
            and shown --, min/max_clusters, validation).
        dendrogram_opts: figure configuration (reused from the fixed-k
            dendrogram figure for visual consistency).
        save_html_to: if provided, writes HTML there (e.g., "out/inconsistency.html").
        export: HTML export options.
        selected_rank: which candidate (1 = best, 2 = second-best, ...) to
            return as the primary ``labels`` output. All candidates are
            still computed and shown in the figure/report regardless of
            this choice; it only picks which one becomes the "main"
            result. Must be between 1 and ``clustering.n_candidates``.
        return_fig: if True also returns the plotly Figure.
        return_html: if True also returns the HTML string.
        return_report: if True also returns the ranked inconsistency report
            (list of dicts, see ``compute_inconsistency_clustering``) --
            this is the simplest way to inspect/adopt every candidate
            (each entry already carries its own ``labels``).

    Returns:
        By default returns cluster labels (np.ndarray) for ``selected_rank``.
        Additional outputs (fig, html, report) are appended, in that order,
        for each corresponding ``return_*`` flag set to True. If more than
        one flag is set, a tuple is returned; otherwise just the labels.
    """
    try:
        Z, _, _, report, R = compute_inconsistency_clustering(
            distance_matrix=distance_matrix,
            genes=genes,
            options=clustering,
        )

        if not (1 <= selected_rank <= len(report)):
            raise ValueError(
                f"selected_rank={selected_rank} is out of range; "
                f"only {len(report)} candidate(s) were computed "
                f"(clustering.n_candidates={clustering.n_candidates})."
            )

        labels = report[selected_rank - 1]["labels"]

        fig: Optional[go.Figure] = None
        html: Optional[str] = None

        # Only build the figure if needed (save or return)
        if save_html_to is not None or return_fig or return_html:
            fig = _build_inconsistency_figure(
                Z=Z,
                genes=_validate_genes(genes, distance_matrix.shape[0]),
                R=R,
                candidates=report,
                opts=dendrogram_opts,
            )
            html = _figure_to_html(fig, export)

            if save_html_to is not None:
                out = _write_text(save_html_to, html)
                _log_or_print(
                    f"[he_inconsistency_clustering] Figure HTML saved at: {out} "
                    f"({len(report)} candidate(s) shown)",
                    export.verbose,
                )

        # Return variants, built dynamically so any combination of flags works
        outputs: list = [labels]
        if return_fig:
            outputs.append(fig)
        if return_html:
            outputs.append(html)
        if return_report:
            outputs.append(report)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    except Exception as e:
        logger.exception("Error in he_inconsistency_clustering")
        raise RuntimeError(f"he_inconsistency_clustering failed: {e}") from e
