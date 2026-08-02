"""
heatmaps: Single-matrix clustered heatmap built on HoloViews + Bokeh.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses             import dataclass
from pathlib                 import Path
from typing                  import Optional, Literal, Union, Tuple, List
from scipy.cluster.hierarchy import linkage, dendrogram, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance  import squareform
import logging
import numpy as np

logger = logging.getLogger(__name__)

PathLike       = Union[str, Path]
DownsampleMode = Literal["none", "pool_mean", "pool_max"]
LinkageMethod  = Literal["average", "complete", "single", "ward", "weighted", "centroid", "median"]


# ──────────────────────────────────────────────────────────────────────────────
# Options
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HeatmapExportOptions:
    """
    Export options for saving an HTML plot.

    Attributes:
        resources: "cdn" loads Bokeh's JS from a CDN (small file, needs
            internet to view); "inline" embeds Bokeh's JS in the file
            (larger, fully standalone/offline).
        verbose: If True, prints extra status messages (still logs always).
    """
    resources: Literal["cdn", "inline"] = "cdn"
    verbose: bool                       = True


@dataclass(frozen=True)
class HeatmapScaleOptions:
    """
    Large-scale options to keep plots usable for big N.

    Attributes:
        max_dim: If matrix is larger than this in any dimension, it will be downsampled.
        downsample_mode: pooling strategy ("pool_mean" recommended).
    """
    max_dim: int                    = 1200
    downsample_mode: DownsampleMode = "pool_mean"


@dataclass(frozen=True)
class ClusteringOptions:
    """
    Hierarchical clustering options for the clustered heatmap.

    The input matrix is assumed to be a **similarity** matrix (values in
    [0, 1]), so distance is computed internally as ``1 - matrix``.

    Attributes:
        cluster_rows: Whether to reorder rows by hierarchical clustering.
        cluster_cols: Whether to reorder columns by hierarchical clustering.
        linkage_method: Linkage algorithm passed to scipy.cluster.hierarchy.linkage.
        optimal_leaf_order: Apply Optimal Leaf Ordering for visually cleaner
            dendrograms (adds O(n^2) cost).
        show_row_dendrogram: Render the row dendrogram as a left side panel.
        show_col_dendrogram: Render the column dendrogram as a top panel.
        dendrogram_fraction: Fraction (in pixels, via frame size) each
            dendrogram panel takes relative to the heatmap.
        dendrogram_line_color: Colour of dendrogram branches.
        dendrogram_line_width: Stroke width of dendrogram branches (px).
    """
    cluster_rows: bool = True
    cluster_cols: bool = True
    linkage_method: LinkageMethod = "average"
    optimal_leaf_order: bool = True
    show_row_dendrogram: bool = True
    show_col_dendrogram: bool = True
    dendrogram_fraction: float = 0.18
    dendrogram_line_color: str = "#4a4a6a"
    dendrogram_line_width: float = 1.5


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log_or_print(msg: str, verbose: bool) -> None:
    logger.info(msg)
    if verbose:
        print(msg)


def _as_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _validate_matrix_2d(matrix: np.ndarray, name: str = "matrix") -> None:
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got: {type(matrix)}")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D, got ndim={matrix.ndim}")
    if matrix.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"{name} must be numeric dtype, got: {matrix.dtype}")


def _pool_downsample(
    matrix: np.ndarray, max_dim: int, mode: DownsampleMode
) -> Tuple[np.ndarray, int, int]:
    """Downsample by block pooling to keep plots responsive."""
    if mode == "none":
        return matrix, 1, 1

    h, w = matrix.shape
    if h <= max_dim and w <= max_dim:
        return matrix, 1, 1

    pool_h = int(np.ceil(h / max_dim))
    pool_w = int(np.ceil(w / max_dim))

    pad_h = (pool_h - (h % pool_h)) % pool_h
    pad_w = (pool_w - (w % pool_w)) % pool_w

    padded = np.pad(
        matrix.astype(np.float64, copy=False),
        ((0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=np.nan,
    )

    new_h = padded.shape[0] // pool_h
    new_w = padded.shape[1] // pool_w
    reshaped = padded.reshape(new_h, pool_h, new_w, pool_w)

    if mode == "pool_mean":
        reduced = np.nanmean(reshaped, axis=(1, 3))
    elif mode == "pool_max":
        reduced = np.nanmax(reshaped, axis=(1, 3))
    else:
        raise ValueError(f"Unsupported downsample mode: {mode}")

    return reduced, pool_h, pool_w


def _similarity_to_linkage(
    sim_matrix: np.ndarray, method: str, use_olo: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a square similarity matrix to a scipy linkage matrix + leaf order."""
    dist = np.clip(1.0 - sim_matrix.astype(np.float64), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method=method)
    if use_olo:
        Z = optimal_leaf_ordering(Z, condensed)

    order = leaves_list(Z)
    return Z, order


def _dendrogram_paths(Z: np.ndarray, orientation: Literal["top", "left"]) -> List[list]:
    """
    Build a list of polylines (one per branch) describing a dendrogram, in
    the same [0, n-1] leaf coordinate system used by the heatmap.
    """
    ddata = dendrogram(Z, no_plot=True)
    icoord = (np.array(ddata["icoord"]) - 5) / 10.0  # -> [0, n-1] leaf axis
    dcoord = np.array(ddata["dcoord"])                # height axis

    paths = []
    for xs, ys in zip(icoord, dcoord):
        if orientation == "top":
            paths.append(list(zip(xs, ys)))
        else:  # "left": mirror so the tree opens toward the heatmap
            paths.append(list(zip([-y for y in ys], xs)))
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def plot_clustered_heatmap(
    matrix: np.ndarray,
    labels: Optional[list] = None,
    save_filepath: Optional[PathLike] = "clustered_heatmap.html",
    x_label: str = "",
    y_label: str = "",
    title: str = "Clustered Heatmap",
    colorscale: str = "RdYlBu",
    z_label: str = "Similarity",
    export: HeatmapExportOptions = HeatmapExportOptions(),
    scale: HeatmapScaleOptions = HeatmapScaleOptions(),
    clustering: ClusteringOptions = ClusteringOptions(),
    heatmap_size: int = 500,
    return_fig: bool = False,
    return_html: bool = False,
):
    """
    Create an interactive clustered heatmap (HoloViews + Bokeh backend) with
    row/column dendrograms whose zoom/pan is natively linked to the heatmap.

    The matrix is assumed to be a square, symmetric **similarity** matrix
    (values in [0, 1], e.g. Jaccard/Wang/cosine similarity). Distance is
    computed internally as ``1 - similarity`` for clustering.

    Parameters
    ----------
    matrix : np.ndarray
        Square 2-D similarity matrix.
    labels : list, optional
        Tick labels for rows/columns (length == matrix.shape[0]).
        Dropped automatically when downsampling is applied.
    save_filepath : PathLike or None
        Destination HTML file (a standalone document). Pass None to skip saving.
    x_label, y_label : str
        Axis titles for the heatmap panel.
    title : str
        Heatmap panel title.
    colorscale : str
        Bokeh/HoloViews colormap name (e.g. "RdYlBu", "Viridis").
    z_label : str
        Colorbar label.
    export : HeatmapExportOptions
        HTML export settings (CDN vs inline Bokeh JS).
    scale : HeatmapScaleOptions
        Downsampling settings for very large matrices.
    clustering : ClusteringOptions
        Hierarchical clustering and dendrogram settings.
    heatmap_size : int
        Heatmap panel size in pixels (square). Dendrogram panel thickness is
        derived from ``clustering.dendrogram_fraction``.
    return_fig : bool
        If True, return the HoloViews Layout/Element object.
    return_html : bool
        If True, return the standalone HTML string.

    Returns
    -------
    None | object | str | (object, str)
        Depends on return_fig / return_html flags.
    """
    import holoviews as hv
    from bokeh.models import HoverTool

    hv.extension("bokeh")

    _validate_matrix_2d(matrix, "matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("plot_clustered_heatmap_hv requires a square matrix.")

    n_orig = matrix.shape[0]
    if labels is not None and len(labels) != n_orig:
        raise ValueError(
            f"labels length ({len(labels)}) must match matrix rows ({n_orig})."
        )
    tick_labels = labels if labels is not None else [str(i) for i in range(n_orig)]

    # ── 1. Downsample if needed ──────────────────────────────────────────────
    z, pool_h, pool_w = _pool_downsample(matrix, scale.max_dim, scale.downsample_mode)
    if pool_h > 1 or pool_w > 1:
        _log_or_print(
            f"[clustered_heatmap_hv] Downsampled {matrix.shape} -> {z.shape} "
            f"(pool={pool_h}x{pool_w}, mode={scale.downsample_mode}).",
            export.verbose,
        )
        tick_labels = [str(i) for i in range(z.shape[0])]
        working_matrix = z
    else:
        working_matrix = matrix

    n = working_matrix.shape[0]
    if n < 2:
        raise ValueError("plot_clustered_heatmap_hv requires at least 2 rows/cols.")

    # ── 2. Clustering (rows and columns can be reordered independently) ─────
    row_order = np.arange(n)
    col_order = np.arange(n)
    row_Z = col_Z = None

    if clustering.cluster_rows:
        row_Z, row_order = _similarity_to_linkage(
            working_matrix, clustering.linkage_method, clustering.optimal_leaf_order
        )
        _log_or_print(
            f"[clustered_heatmap_hv] Row clustering done "
            f"(method={clustering.linkage_method}, OLO={clustering.optimal_leaf_order}).",
            export.verbose,
        )
    if clustering.cluster_cols:
        col_Z, col_order = _similarity_to_linkage(
            working_matrix, clustering.linkage_method, clustering.optimal_leaf_order
        )
        _log_or_print(
            f"[clustered_heatmap_hv] Column clustering done "
            f"(method={clustering.linkage_method}, OLO={clustering.optimal_leaf_order}).",
            export.verbose,
        )

    z_clustered = working_matrix[np.ix_(row_order, col_order)]
    row_labels = [tick_labels[i] for i in row_order]
    col_labels = [tick_labels[i] for i in col_order]

    has_col_dend = clustering.cluster_cols and clustering.show_col_dendrogram and col_Z is not None
    has_row_dend = clustering.cluster_rows and clustering.show_row_dendrogram and row_Z is not None

    # ── 3. Heatmap element ───────────────────────────────────────────────────
    records = []
    for i in range(n):
        for j in range(n):
            records.append((j, i, float(z_clustered[i, j]), row_labels[i], col_labels[j]))

    hover = HoverTool(tooltips=[
        ("Row", "@row_label"),
        ("Col", "@col_label"),
        (z_label, "@similarity{0.000}"),
    ])

    heat = hv.HeatMap(
        records, kdims=["leaf_x", "leaf_y"], vdims=["similarity", "row_label", "col_label"]
    ).opts(
        cmap=colorscale + "_r" if not colorscale.endswith("_r") else colorscale,
        colorbar=True,
        clim=(0.0, 1.0),
        colorbar_opts={"title": z_label},
        frame_width=heatmap_size,
        frame_height=heatmap_size,
        tools=[hover],
        xticks=list(enumerate(col_labels)),
        yticks=list(enumerate(row_labels)),
        xrotation=45,
        xlabel=x_label,
        ylabel=y_label,
        title=title,
        line_color=None,
    )

    elements = []
    dend_size = max(60, int(heatmap_size * clustering.dendrogram_fraction))

    # ── 4. Dendrogram elements (leaf axis shares the SAME dimension name as
    #      the heatmap, which is what makes HoloViews link their ranges) ─────
    col_dend = None
    if has_col_dend:
        col_paths = _dendrogram_paths(col_Z, "top")
        col_dend = hv.Path(col_paths, kdims=["leaf_x", "height_col"]).opts(
            color=clustering.dendrogram_line_color,
            line_width=clustering.dendrogram_line_width,
            frame_width=heatmap_size,
            frame_height=dend_size,
            xaxis=None,
            yaxis=None,
            title="",
        )

    row_dend = None
    if has_row_dend:
        row_paths = _dendrogram_paths(row_Z, "left")
        row_dend = hv.Path(row_paths, kdims=["height_row", "leaf_y"]).opts(
            color=clustering.dendrogram_line_color,
            line_width=clustering.dendrogram_line_width,
            frame_width=dend_size,
            frame_height=heatmap_size,
            xaxis=None,
            yaxis=None,
            title="",
        )

    # ── 5. Compose grid layout ───────────────────────────────────────────────
    if col_dend is not None and row_dend is not None:
        blank = hv.Empty()
        layout = (blank + col_dend + row_dend + heat).cols(2)
    elif col_dend is not None:
        layout = (col_dend + heat).cols(1)
    elif row_dend is not None:
        layout = (row_dend + heat).cols(2)
    else:
        layout = heat

    layout = layout.opts(hv.opts.Layout(shared_axes=True)) if hasattr(layout, "opts") and not isinstance(layout, hv.HeatMap) else layout

    # ── 6. Export ────────────────────────────────────────────────────────────
    html: Optional[str] = None
    if save_filepath is not None or return_html:
        renderer = hv.renderer("bokeh")
        resources = "cdn" if export.resources == "cdn" else "inline"
        html = renderer.html(layout, resources=resources)
        if save_filepath is not None:
            p = _as_path(save_filepath)
            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(html, encoding="utf-8")
            _log_or_print(f"[clustered_heatmap_hv] HTML saved at: {p}", export.verbose)

    if return_fig and return_html:
        return layout, html
    if return_fig:
        return layout
    if return_html:
        return html
    return None