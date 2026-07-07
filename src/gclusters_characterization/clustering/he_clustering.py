"""
HeClustering

This module provides utilities for hierarchical clustering using a
distance matrix representation of elements (e.g., genes).

The module supports:
- Computation of hierarchical clustering using SciPy linkage methods.
- Generation of flat clusters via the maxclust criterion.
- Optional creation of interactive dendrogram visualizations using Plotly.
- Export of dendrogram figures to HTML format.

Functions
1. compute_hierarchical_clustering – Perform hierarchical clustering and compute cluster labels.
2. compute_dynamic_clustering      – Automatically detect the number of clusters from the linkage matrix.
3. he_clustering – High-level interface for clustering, visualization, and export.
4. _validate_distance_matrix – Validate structural properties of a distance matrix.
5. _validate_genes – Validate gene labels associated with the distance matrix.
6. _build_dendrogram_figure – Construct an interactive dendrogram figure.
7. _figure_to_html – Convert a Plotly figure into HTML.
8. _write_text – Write HTML output to disk.
9. _detect_gap_cut – Detect the optimal cut point from the largest height gap in Z.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from pathlib import Path
from typing import  List, Optional, Sequence, Tuple, Union, Literal
import logging
import sys
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import squareform
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data Types
# ──────────────────────────────────────────────────────────────────────────────
PathLike = Union[str, Path]
LinkageMethod = Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"]


# ──────────────────────────────────────────────────────────────────────────────
# Classes
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ClusteringOptions:
    """
    Options for hierarchical clustering.

    Attributes:
        num_groups: number of clusters for fcluster(maxclust). Must be >= 1.
        method: linkage method (scipy).
        validate_distance: enable strict distance-matrix checks.
        sym_tol: tolerance used in symmetry checks (np.allclose).
    """
    num_groups: int = 4
    method: LinkageMethod = "single"
    validate_distance: bool = True
    sym_tol: float = 1e-12
    verbose: bool = True


@dataclass(frozen=True)
class DynamicClusteringOptions:
    """
    Options for automatic (gap-based) hierarchical clustering.

    Instead of requiring the user to specify ``num_groups``, this approach
    analyses the linkage matrix ``Z`` to find the merge step with the largest
    jump in fusion height (the *largest gap*).  Cutting just below that jump
    yields the most natural number of clusters for the data.

    Attributes:
        method : LinkageMethod
            Linkage algorithm passed to ``scipy.cluster.hierarchy.linkage``.
        n_gaps : int
            Number of top gaps to consider as candidate cut points.
            When ``n_gaps=1`` (default) the single largest gap is used.
            Increasing it lets you inspect a ranked list of alternatives
            via the ``gap_report`` returned by ``compute_dynamic_clustering``.
        min_clusters : int
            Lower bound on the number of clusters accepted.  Any gap that
            would produce fewer clusters than this is skipped.  Default 2.
        max_clusters : int or None
            Upper bound on the number of clusters.  Any gap that would
            produce more clusters than this is skipped.  ``None`` = no limit.
        validate_distance : bool
            Enable strict distance-matrix structural checks.
        sym_tol : float
            Numerical tolerance used in symmetry and diagonal checks.
        verbose : bool
            If ``True``, prints status messages (always logs regardless).
    """
    method: LinkageMethod = "average"
    n_gaps: int = 1
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


def _validate_genes(genes: Sequence[str], n: int) -> List[str]:
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
        raise ValueError(f"Number of genes ({len(genes)}) does not match matrix size ({n}).")

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
        raise TypeError(f"distance_matrix must be a numpy.ndarray, got: {type(distance_matrix)}")

    if distance_matrix.ndim != 2:
        raise ValueError(f"distance_matrix must be 2D, got ndim={distance_matrix.ndim}")

    n, m = distance_matrix.shape
    if n != m:
        raise ValueError("Distance matrix must be square.")

    if n < 2:
        raise ValueError("Distance matrix must be at least 2x2.")

    if not np.issubdtype(distance_matrix.dtype, np.number):
        raise TypeError(f"distance_matrix must be numeric dtype, got: {distance_matrix.dtype}")

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


def _compute_cutoff_height(Z: np.ndarray, num_groups: int) -> Optional[float]:
    """
    Compute a visualization cutoff height for the dendrogram.

    This cutoff is used only for visualization purposes and does not
    affect clustering assignments.

    Parameters
    ----------
    Z : numpy.ndarray
        Linkage matrix produced by SciPy hierarchical clustering.
    num_groups : int
        Number of clusters requested by the user.

    Returns
    -------
    float or None
        Height at which a horizontal cutoff line should be drawn,
        or None if no cutoff is appropriate.
    """
    if num_groups <= 1:
        return None
    if Z.shape[0] == 0:
        return None
    idx = -(num_groups - 1)
    if abs(idx) > Z.shape[0]:
        # num_groups is too large relative to n, fallback to minimum possible cutoff
        return float(np.min(Z[:, 2]))
    return float(Z[idx, 2])


def _build_dendrogram_figure(
    Z: np.ndarray,
    genes: Sequence[str],
    num_groups: int,
    opts: DendrogramOptions,
) -> go.Figure:
    """
    Construct an interactive Plotly dendrogram figure.
    """

    # Labels with stable index prefix
    labels = [f"{i}-{g}" for i, g in enumerate(genes)]

    # Compute cutoff height used for coloring
    cutoff_height = _compute_cutoff_height(Z, num_groups)

    # Compute dendrogram structure.
    # scipy.cluster.hierarchy.dendrogram is recursive; large linkage trees
    # can exceed Python's default limit (~1000 frames). We raise the limit
    # temporarily and restore it afterwards.
    _prev_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(_prev_limit, len(Z) * 3 + 1000))
    try:
        ddata = dendrogram(
            Z,
            labels=labels,
            color_threshold=cutoff_height,
            above_threshold_color=opts.line_color,
            no_plot=True
        )
    finally:
        sys.setrecursionlimit(_prev_limit)

    fig = go.Figure()

    # Get matplotlib default color cycle
    mpl_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Add dendrogram segments
    for x, y, color in zip(
        ddata["icoord"],
        ddata["dcoord"],
        ddata["color_list"]
    ):

        # Convert matplotlib shorthand colors (C0, C1, ...)
        try:
            if isinstance(color, str) and color.startswith("C"):
                idx = int(color[1:])
                color = mpl_cycle[idx % len(mpl_cycle)]

            # Convert to hex for plotly compatibility
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
                line=dict(color=color),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Leaf ordering
    leaf_idx = ddata["leaves"]
    leaf_labels = [labels[i] for i in leaf_idx]
    leaf_positions = [5 + 10 * i for i in range(len(leaf_idx))]

    # Optional cutoff line
    if opts.show_cutoff and cutoff_height is not None:

        x_range = [0, 10 * len(leaf_idx)]

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[cutoff_height, cutoff_height],
                mode="lines",
                line=dict(
                    color=opts.cutoff_color,
                    dash=opts.cutoff_dash
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Title
    title = opts.title
    if num_groups > 1:
        title = f"{opts.title} ({num_groups} clusters)"

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Genes (Index-Name)",
            tickmode="array",
            tickvals=leaf_positions,
            ticktext=leaf_labels,
            tickangle=opts.tick_angle,
        ),
        yaxis=dict(title="Distance"),
        height=opts.height,
        width=opts.width,
        hovermode="closest",
    )

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
    return fig.to_html(include_plotlyjs=export.include_plotlyjs, full_html=export.full_html)


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

def _detect_gap_cut(
    Z: np.ndarray,
    n_gaps: int,
    min_clusters: int,
    max_clusters: Optional[int],
) -> List[Tuple[int, float, float]]:
    """
    Identify candidate cut points in a linkage matrix by largest height gap.

    The linkage matrix ``Z`` contains one row per merge step; column 2 stores
    the fusion height at each step.  The difference between consecutive heights
    (*gaps*) reveals how dissimilar the elements being merged are.  A large gap
    means the algorithm was forced to join two relatively distant groups —
    exactly where a natural cluster boundary lies.

    Parameters
    ----------
    Z : numpy.ndarray
        Linkage matrix (n-1 × 4) from ``scipy.cluster.hierarchy.linkage``.
    n_gaps : int
        How many top-gap candidates to return (ranked by gap size, descending).
    min_clusters : int
        Minimum number of clusters a candidate must yield to be included.
    max_clusters : int or None
        Maximum number of clusters allowed. ``None`` = no upper limit.

    Returns
    -------
    List[Tuple[int, float, float]]
        Each entry is ``(k, gap_size, cut_height)`` where:

        - ``k``           — number of clusters obtained by cutting here.
        - ``gap_size``    — magnitude of the height jump (larger = more natural).
        - ``cut_height``  — height threshold at which the tree is cut
                            (midpoint of the gap, safe for ``fcluster``).

    Notes
    -----
    Cutting at the *midpoint* of the gap avoids numerical edge cases where
    cutting exactly at the lower bound would include the merge itself.
    """
    heights = Z[:, 2]

    # gaps[i] = height[i+1] - height[i]  (positive because Z is sorted)
    gaps = np.diff(heights)

    # Number of clusters when cutting between step i and i+1 is (n - i - 1)
    # because steps 0..i have already been merged into (n-i-1) groups.
    n_steps = len(heights)          # == n - 1
    n_leaves = n_steps + 1          # original number of elements

    # Build candidate list: (gap_index_in_gaps, k_clusters, gap_size)
    candidates = []
    for i in range(len(gaps)):
        k = n_leaves - (i + 1)     # clusters *after* merges 0..i
        if k < min_clusters:
            continue
        if max_clusters is not None and k > max_clusters:
            continue
        candidates.append((i, k, float(gaps[i])))

    if not candidates:
        raise ValueError(
            f"No valid cut point found with min_clusters={min_clusters} "
            f"and max_clusters={max_clusters}.  Try relaxing these bounds."
        )

    # Sort by gap size descending, take top n_gaps
    candidates.sort(key=lambda x: x[2], reverse=True)
    top = candidates[:n_gaps]

    result = []
    for gap_idx, k, gap_size in top:
        # Midpoint of the gap for a safe fcluster threshold
        cut_height = float((heights[gap_idx] + heights[gap_idx + 1]) / 2.0)
        result.append((k, gap_size, cut_height))

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main Functions
# ──────────────────────────────────────────────────────────────────────────────
def compute_hierarchical_clustering(
    distance_matrix: np.ndarray,
    genes: Sequence[str],
    options: ClusteringOptions = ClusteringOptions(),
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform hierarchical clustering and evaluate dendrogram quality.

    In addition to computing cluster assignments, this function
    calculates the cophenetic correlation coefficient which measures
    how faithfully the dendrogram preserves the original distances.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Square pairwise distance matrix (n x n).

    genes : Sequence[str]
        Gene identifiers corresponding to matrix rows.

    options : ClusteringOptions
        Configuration parameters for hierarchical clustering.

    Returns
    -------
    Z : numpy.ndarray
        Linkage matrix describing the hierarchical clustering.

    labels : numpy.ndarray
        Cluster label assigned to each gene.

    cophenetic_corr : float
        Cophenetic correlation coefficient measuring how well the
        dendrogram preserves the original pairwise distances.
    """

    if options.num_groups < 1:
        raise ValueError("num_groups must be >= 1.")

    if options.validate_distance:
        _validate_distance_matrix(distance_matrix, tol=options.sym_tol)

    n = distance_matrix.shape[0]
    genes = _validate_genes(genes, n)

    # Convert to condensed form
    condensed = squareform(distance_matrix, checks=False)

    # Compute hierarchical clustering
    Z = linkage(condensed, method=options.method)

    # Compute flat clusters
    labels = fcluster(Z, options.num_groups, criterion="maxclust")

    # Compute cophenetic correlation
    cophenetic_corr, _ = cophenet(Z, condensed)

    # information logging.
    interpretation = (
        "strong" if cophenetic_corr > 0.75
        else "moderate" if cophenetic_corr > 0.5
        else "weak"
    )

    _log_or_print(
        f"Cophenetic correlation coefficient: {cophenetic_corr:.4f} " +
        f"({interpretation} clustering structure)",
        verbose=options.verbose
    )

    return Z, labels, cophenetic_corr




def compute_dynamic_clustering(
    distance_matrix: np.ndarray,
    genes: "Sequence[str]",
    options: "DynamicClusteringOptions" = None,
) -> "Tuple[np.ndarray, np.ndarray, float, List[dict]]":
    """
    Perform hierarchical clustering with automatic cluster-count detection.

    Unlike ``compute_hierarchical_clustering``, this function does **not**
    require the user to specify ``num_groups``. Instead it examines the
    linkage matrix to find the merge step with the largest jump in fusion
    height -- the *gap heuristic* -- and cuts the tree there.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Square pairwise distance matrix (n x n), values in [0, inf).
        The diagonal must be zero and the matrix must be symmetric.
    genes : Sequence[str]
        Gene identifiers corresponding to matrix rows/columns.
    options : DynamicClusteringOptions
        Configuration for clustering method, gap search, and constraints.

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
    gap_report : List[dict]
        Ranked list of the top-n_gaps candidate cut points, each with:

        - 'rank'        -- 1 = best (largest gap).
        - 'k'           -- number of clusters at this cut.
        - 'gap_size'    -- height jump magnitude.
        - 'cut_height'  -- distance threshold used for fcluster.
        - 'selected'    -- True only for the chosen cut (rank 1).

    Raises
    ------
    ValueError
        If the distance matrix is invalid, gene list length is wrong, or no
        valid cut point exists within the min/max_clusters constraints.

    Examples
    --------
    >>> import numpy as np
    >>> from he_clustering import compute_dynamic_clustering, DynamicClusteringOptions
    >>>
    >>> dist = np.array([[0,   0.2, 0.9, 0.8],
    ...                  [0.2, 0,   0.85,0.75],
    ...                  [0.9, 0.85,0,   0.15],
    ...                  [0.8, 0.75,0.15,0   ]])
    >>> genes = ["GeneA", "GeneB", "GeneC", "GeneD"]
    >>>
    >>> Z, labels, coph, report = compute_dynamic_clustering(dist, genes)
    >>> print(labels)        # e.g. [1 1 2 2]
    >>> print(report[0])     # best cut details
    """
    if options is None:
        options = DynamicClusteringOptions()

    if options.validate_distance:
        _validate_distance_matrix(distance_matrix, tol=options.sym_tol)

    n = distance_matrix.shape[0]
    genes_validated = _validate_genes(genes, n)

    if n < 3:
        raise ValueError(
            "Dynamic clustering requires at least 3 elements to compute gaps."
        )

    # Build linkage
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method=options.method)

    # Cophenetic correlation
    cophenetic_corr, _ = cophenet(Z, condensed)

    interpretation = (
        "strong" if cophenetic_corr > 0.75
        else "moderate" if cophenetic_corr > 0.5
        else "weak"
    )
    _log_or_print(
        f"[dynamic] Cophenetic correlation: {cophenetic_corr:.4f} ({interpretation})",
        options.verbose,
    )

    # Detect gap-based cut points
    candidates = _detect_gap_cut(
        Z,
        n_gaps=options.n_gaps,
        min_clusters=options.min_clusters,
        max_clusters=options.max_clusters,
    )

    # Best cut is the first (largest gap)
    best_k, best_gap, best_cut_height = candidates[0]

    _log_or_print(
        f"[dynamic] Largest gap: {best_gap:.6f} -- cutting at height "
        f"{best_cut_height:.6f} -> {best_k} clusters.",
        options.verbose,
    )

    # Assign flat clusters using the detected height threshold
    labels = fcluster(Z, t=best_cut_height, criterion="distance")

    # Build gap report
    gap_report = []
    for rank, (k, gap_size, cut_height) in enumerate(candidates, start=1):
        gap_report.append({
            "rank":       rank,
            "k":          k,
            "gap_size":   gap_size,
            "cut_height": cut_height,
            "selected":   rank == 1,
        })

    if options.verbose and len(gap_report) > 1:
        _log_or_print(
            "[dynamic] Alternative cut points (ranked by gap size):",
            options.verbose,
        )
        for entry in gap_report[1:]:
            _log_or_print(
                f"  rank={entry['rank']}  k={entry['k']}  "
                f"gap={entry['gap_size']:.6f}  height={entry['cut_height']:.6f}",
                options.verbose,
            )

    return Z, labels, cophenetic_corr, gap_report

def he_clustering(
    distance_matrix: np.ndarray,
    genes: Sequence[str],
    clustering: ClusteringOptions = ClusteringOptions(),
    dendrogram_opts: DendrogramOptions = DendrogramOptions(),
    save_html_to: Optional[PathLike] = None,
    export: ExportOptions = ExportOptions(),
    return_fig: bool = False,
    return_html: bool = False,
):
    """
    High-level function:
    - Computes hierarchical clustering
    - Optionally builds an interactive dendrogram (plotly)
    - Optionally exports to HTML

    Args:
        distance_matrix: square distance matrix (n x n).
        genes: gene identifiers (len n).
        clustering: clustering configuration (method, num_groups, validation).
        dendrogram_opts: figure configuration.
        save_html_to: if provided, writes HTML there (e.g., "out/dendrogram.html").
        export: HTML export options.
        return_fig: if True returns plotly Figure.
        return_html: if True returns HTML string.

    Returns:
        By default returns cluster labels (np.ndarray).
        If return_fig=True -> returns (labels, fig)
        If return_html=True -> returns (labels, html)
        If both True -> returns (labels, fig, html)
    """
    try:
        Z, labels, _ = compute_hierarchical_clustering(
            distance_matrix=distance_matrix,
            genes=genes,
            options=clustering,
        )

        fig: Optional[go.Figure] = None
        html: Optional[str] = None

        # Only build the figure if needed (save or return)
        if save_html_to is not None or return_fig or return_html:
            fig = _build_dendrogram_figure(
                Z=Z,
                genes=_validate_genes(genes, distance_matrix.shape[0]),
                num_groups=clustering.num_groups,
                opts=dendrogram_opts,
            )
            html = _figure_to_html(fig, export)

            if save_html_to is not None:
                out = _write_text(save_html_to, html)
                _log_or_print(f"[he_clustering] Dendrogram HTML saved at: {out}", export.verbose)

        # Return variants
        if return_fig and return_html:
            return labels, fig, html
        if return_fig:
            return labels, fig
        if return_html:
            return labels, html
        return labels

    except Exception as e:
        logger.exception("Error in he_clustering")
        # Keep a clean public error message
        raise RuntimeError(f"he_clustering failed: {e}") from e


__all__ = [
    "ClusteringOptions",
    "DynamicClusteringOptions",
    "DendrogramOptions",
    "ExportOptions",
    "compute_hierarchical_clustering",
    "compute_dynamic_clustering",
    "he_clustering",
]