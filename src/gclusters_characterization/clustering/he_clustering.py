"""
Hierarchical clustering utilities with optional interactive dendrogram export.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass                                           # Decorator to automatically generate special methods (e.g., __init__).
from pathlib import Path                                                    # Object-oriented filesystem path handling.
from typing import  List, Optional, Sequence, Tuple, Union, Literal         # Improve type hints and function signatures.
import logging                                                              # Advanced logging system for status and error messages.
import numpy as np                                                          # Efficient numerical computations.
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster           # Clustering functions.
from scipy.spatial.distance import squareform                               # Distance matrix for clustering.
import plotly.graph_objects as go                                           # Plotting graphs.


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)                                        # Initialize module-level logger.

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
    Library-friendly output handler:
    - Always logs the message using the module logger.
    - Optionally prints the message if verbose=True.
    """
    logger.info(msg)
    if verbose:
        print(msg)


def _validate_genes(genes: Sequence[str], n: int) -> List[str]:
    if not isinstance(genes, (list, tuple)):
        genes = list(genes)

    if len(genes) != n:
        raise ValueError(f"Number of genes ({len(genes)}) does not match matrix size ({n}).")

    # Ensure all are strings (best-effort)
    out = [str(g) for g in genes]
    return out


def _validate_distance_matrix(distance_matrix: np.ndarray, tol: float) -> None:
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
    Compute a cutoff height for visualization only.

    For maxclust:
    - If num_groups <= 1: no cutoff line is meaningful.
    - Otherwise, a common heuristic is to use the merge height that yields that many clusters.
      We use Z[-(num_groups-1), 2] but guarded for safety.
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
    # Labels with stable index prefix
    labels = [f"{i}-{g}" for i, g in enumerate(genes)]

    # Compute dendrogram structure without plotting
    ddata = dendrogram(Z, labels=labels, no_plot=True)

    fig = go.Figure()

    # Add dendrogram segments
    # SciPy gives icoord/dcoord; each entry is length-4 polyline.
    for x, y in zip(ddata["icoord"], ddata["dcoord"]):
        # We break into segments with None separators for plotly
        xs = [x[0], x[1], None, x[1], x[2], None, x[2], x[3], None]
        ys = [y[0], y[1], None, y[1], y[2], None, y[2], y[3], None]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=opts.line_color),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Leaf labels and positions (Scipy's convention: 5, 15, 25, ...)
    leaf_idx = ddata["leaves"]
    leaf_labels = [labels[i] for i in leaf_idx]
    leaf_positions = [5 + 10 * i for i in range(len(leaf_idx))]

    # Optional cutoff line
    cutoff_height = _compute_cutoff_height(Z, num_groups)
    if opts.show_cutoff and cutoff_height is not None:
        x_range = [0, 10 * len(leaf_idx)]
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[cutoff_height, cutoff_height],
                mode="lines",
                line=dict(color=opts.cutoff_color, dash=opts.cutoff_dash),
                hoverinfo="skip",
                showlegend=False,
            )
        )

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
    """If it is stablish by user, the plot is completely HTML"""
    return fig.to_html(include_plotlyjs=export.include_plotlyjs, full_html=export.full_html)


def _write_text(filepath: PathLike, content: str) -> Path:
    """Write file into a Path"""
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute hierarchical clustering (linkage + flat clusters).

    Returns:
        Z: linkage matrix
        labels: 1D array with cluster assignment per gene
    """
    if options.num_groups < 1:
        raise ValueError("num_groups must be >= 1.")

    if options.validate_distance:
        _validate_distance_matrix(distance_matrix, tol=options.sym_tol)

    n = distance_matrix.shape[0]
    genes = _validate_genes(genes, n)

    # Convert square distance matrix to condensed form
    condensed = squareform(distance_matrix, checks=False)  # checks handled above

    # Linkage
    Z = linkage(condensed, method=options.method)

    # Flat clustering: maxclust
    labels = fcluster(Z, options.num_groups, criterion="maxclust")

    return Z, labels


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
        Z, labels = compute_hierarchical_clustering(
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
