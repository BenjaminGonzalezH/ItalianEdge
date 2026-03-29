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
2. he_clustering – High-level interface for clustering, visualization, and export.
3. _validate_distance_matrix – Validate structural properties of a distance matrix.
4. _validate_genes – Validate gene labels associated with the distance matrix.
5. _build_dendrogram_figure – Construct an interactive dendrogram figure.
6. _figure_to_html – Convert a Plotly figure into HTML.
7. _write_text – Write HTML output to disk.
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
    "DendrogramOptions",
    "ExportOptions",
    "compute_hierarchical_clustering",
    "he_clustering",
]
