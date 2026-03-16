"""
CSPA Method
-----------

Implementation of the Cluster-based Similarity Partitioning Algorithm (CSPA)
for ensemble clustering.

CSPA is a consensus clustering technique that combines multiple clustering
solutions by constructing a co-association (coincidence) matrix representing
how frequently two elements appear in the same cluster.

This module provides:

• Spectral clustering over a coincidence matrix
• Spectral embedding visualization of clustering results
• Interactive Plotly visualization with cluster highlighting
• HTML export of interactive figures

The workflow implemented here follows these steps:

1. Build or receive a coincidence matrix.
2. Apply spectral clustering to obtain consensus labels.
3. Compute a spectral embedding for visualization.
4. Generate an interactive scatter plot.
5. Optionally export the visualization as HTML.

Functions
---------
1. cspa_spectral_from_coincidence
   Perform consensus clustering using spectral clustering.

2. build_click_highlight_embedding_figure
   Generate an interactive embedding visualization.

3. figure_to_html_with_click_highlight
   Export the visualization as HTML with interactive highlighting.

4. plot_embedding_click_highlight
   High-level helper for visualization and export.

5. cspa_method
   End-to-end pipeline for CSPA consensus clustering and visualization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

import logging
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# ──────────────────────────────────────────────────────────────
# Dataclasses (options)
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CSPAOptions:
    """
    Configuration options for the CSPA spectral clustering algorithm.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to produce in the consensus clustering.

    assign_labels : {"kmeans", "discretize", "cluster_qr"}, default="kmeans"
        Method used by sklearn SpectralClustering to assign labels
        after spectral embedding.

        • "kmeans"      : k-means clustering in spectral space.
        • "discretize"  : discretization method proposed in spectral clustering literature.
        • "cluster_qr"  : QR-based cluster assignment.

    random_state : int, default=0
        Random seed used to ensure reproducibility.
    """
    n_clusters: int
    assign_labels: Literal["kmeans", "discretize", "cluster_qr"] = "kmeans"
    random_state: int = 0

@dataclass(frozen=True)
class EmbedOptions:
    """
    Options controlling the spectral embedding used for visualization.

    Spectral embedding projects the affinity matrix into a low-dimensional
    space that preserves graph connectivity structure.

    Parameters
    ----------
    n_components : int, default=2
        Number of embedding dimensions.

    random_state : int, default=0
        Random seed used by the embedding algorithm.
    """
    n_components: int = 2
    random_state: int = 0


@dataclass(frozen=True)
class ExportHTML:
    """
    Configuration options for exporting Plotly figures to HTML.

    Parameters
    ----------
    include_plotlyjs : {"cdn", "embed"} or bool, default="cdn"
        Determines how Plotly JavaScript is included in the HTML file.

        • "cdn"   : loads Plotly from an external CDN (smaller file).
        • "embed" : embeds the full Plotly library inside the HTML.
        • True    : same as "embed".
        • False   : no JavaScript included.

    full_html : bool, default=True
        If True, export a complete standalone HTML document.

    verbose : bool, default=False
        If True, print a message indicating where the HTML file was saved.
    """
    include_plotlyjs: Union[Literal["cdn", "embed"], bool] = "cdn"
    full_html: bool = True
    verbose: bool = False


# ──────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────

def _validate_square_numeric(matrix: np.ndarray, name: str) -> None:
    """
    Validate that a matrix is a numeric square matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix to validate.

    name : str
        Name of the variable used for error messages.

    Raises
    ------
    TypeError
        If the input is not a NumPy array or does not contain numeric values.

    ValueError
        If the matrix is not two-dimensional, empty, or not square.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(matrix)}")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D, got ndim={matrix.ndim}")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square. Got shape={matrix.shape}")
    if matrix.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"{name} must be numeric dtype, got {matrix.dtype}")


def _validate_genes(genes: Sequence[str], n: int) -> None:
    """
    Validate that a gene list matches the number of elements in the matrix.

    Parameters
    ----------
    genes : Sequence[str]
        List of gene identifiers.

    n : int
        Expected number of elements.

    Raises
    ------
    ValueError
        If the gene list is missing or does not match the expected size.
    """
    if genes is None:
        raise ValueError("genes must be provided (Sequence[str]).")
    if len(genes) != n:
        raise ValueError(f"Number of genes ({len(genes)}) does not match n_elements ({n}).")


def _validate_coincidence_matrix(coincidence: np.ndarray, *, tol_sym: float = 1e-8) -> None:
    """
    Validate a coincidence (co-association) matrix used in ensemble clustering.

    The coincidence matrix represents how frequently two elements appear
    in the same cluster across multiple clustering solutions.

    Expected properties
    -------------------
    • square matrix (N x N)
    • numeric values
    • symmetric
    • diagonal values approximately equal to 1
    • finite values (no NaN or Inf)

    Raises
    ------
    ValueError
        If any of the validation conditions fail.
    """
    _validate_square_numeric(coincidence, "coincidence_matrix")

    if not np.all(np.isfinite(coincidence)):
        raise ValueError("coincidence_matrix contains NaN/Inf.")

    if not np.allclose(coincidence, coincidence.T, atol=tol_sym, rtol=0):
        raise ValueError("coincidence_matrix must be symmetric.")

    diag = np.diag(coincidence)
    if not np.allclose(diag, 1.0, atol=1e-6, rtol=0):
        raise ValueError("coincidence_matrix diagonal must be ~1.0 (each item coincides with itself).")


def _as_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _log_or_print(msg: str, verbose: bool) -> None:
    logger.info(msg)
    if verbose:
        print(msg)

# ──────────────────────────────────────────────────────────────
# CSPA (Spectral Clustering)
# ──────────────────────────────────────────────────────────────

def cspa_spectral_from_coincidence(
    coincidence_matrix: np.ndarray,
    options: CSPAOptions,
) -> np.ndarray:
    """
    Perform CSPA consensus clustering using spectral clustering.

    The coincidence matrix represents how often pairs of elements
    appear in the same cluster across multiple clustering solutions.

    This matrix is treated as a similarity (affinity) matrix
    and passed directly to sklearn's SpectralClustering algorithm.

    Parameters
    ----------
    coincidence_matrix : numpy.ndarray
        NxN matrix where entry (i,j) represents the proportion
        of clustering solutions where element i and j appear
        in the same cluster.

    options : CSPAOptions
        Configuration options controlling the clustering process.

    Returns
    -------
    numpy.ndarray
        Vector of cluster labels of length N.

    Raises
    ------
    ValueError
        If the matrix fails validation or cluster configuration is invalid.
    """
    _validate_coincidence_matrix(coincidence_matrix)

    if options.n_clusters < 1:
        raise ValueError("options.n_clusters must be >= 1")
    n = coincidence_matrix.shape[0]
    if options.n_clusters > n:
        raise ValueError("options.n_clusters cannot exceed number of items")

    # Clip tiny numeric drift and ensure float64 for sklearn stability
    A = np.clip(coincidence_matrix.astype(np.float64, copy=False), 0.0, 1.0)

    model = SpectralClustering(
        n_clusters=int(options.n_clusters),
        affinity="precomputed",
        assign_labels=str(options.assign_labels),
        random_state=int(options.random_state),
    )
    return model.fit_predict(A)

def _factorize_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert arbitrary labels into compact integer codes.

    This is useful when cluster labels are strings or
    non-contiguous integers.

    Example
    -------
    labels = ["A", "A", "C"]

    unique_labels = ["A", "C"]
    inverse_codes = [0, 0, 1]

    Parameters
    ----------
    labels : numpy.ndarray
        Array of cluster labels.

    Returns
    -------
    tuple
        unique_labels : numpy.ndarray
            Sorted unique label values.

        inverse_codes : numpy.ndarray
            Integer codes mapping each element to a label index.
    """
    unique, inv = np.unique(labels, return_inverse=True)
    return unique, inv.astype(np.int64, copy=False)

# ──────────────────────────────────────────────────────────────
# Interactive embedding plot (click-to-highlight cluster)
# ──────────────────────────────────────────────────────────────

def _spectral_embedding_2d(
    affinity_matrix: np.ndarray,
    embed: EmbedOptions,
) -> np.ndarray:
    """
    Compute a spectral embedding of the affinity matrix.

    Spectral embedding projects nodes of a graph into a
    lower-dimensional space while preserving connectivity structure.

    Parameters
    ----------
    affinity_matrix : numpy.ndarray
        NxN similarity matrix.

    embed : EmbedOptions
        Embedding configuration parameters.

    Returns
    -------
    numpy.ndarray
        Coordinates of elements in embedding space.
    """
    _validate_square_numeric(affinity_matrix, "affinity_matrix")
    if not np.all(np.isfinite(affinity_matrix)):
        raise ValueError("affinity_matrix contains NaN/Inf.")
    A = affinity_matrix.astype(np.float64, copy=False)

    emb = SpectralEmbedding(
        n_components=int(embed.n_components),
        affinity="precomputed",
        random_state=int(embed.random_state),
    )
    return emb.fit_transform(A)


def build_click_highlight_embedding_figure(
    affinity_matrix: np.ndarray,
    labels: np.ndarray,
    genes: Sequence[str],
    *,
    title: str = "Ensemble Embedding (click to highlight cluster)",
    embed: EmbedOptions = EmbedOptions(),
) -> go.Figure:
    """
    Create an interactive Plotly scatter plot representing the spectral embedding.

    Each cluster is represented as a separate trace so that
    JavaScript can efficiently highlight clusters when clicked.

    Hover information includes:
    • gene identifier
    • cluster label

    Parameters
    ----------
    affinity_matrix : numpy.ndarray
        Matrix used to compute spectral embedding.

    labels : numpy.ndarray
        Cluster labels for each element.

    genes : Sequence[str]
        Gene identifiers corresponding to each element.

    title : str
        Title of the figure.

    embed : EmbedOptions
        Spectral embedding configuration.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive scatter plot.
    """
    _validate_square_numeric(affinity_matrix, "affinity_matrix")
    n = affinity_matrix.shape[0]
    _validate_genes(genes, n)

    labels = np.asarray(labels)
    if labels.shape != (n,):
        raise ValueError(f"labels must have shape (n,), got {labels.shape}")

    coords = _spectral_embedding_2d(affinity_matrix, embed)

    # Factorize labels to stable display/order
    uniq, inv = _factorize_labels(labels)

    fig = go.Figure()
    for k, lab in enumerate(uniq):
        idx = np.where(inv == k)[0]
        # customdata = gene ids to show in hover
        custom = np.array([genes[i] for i in idx], dtype=object)

        fig.add_trace(
            go.Scatter(
                x=coords[idx, 0],
                y=coords[idx, 1],
                mode="markers",
                name=f"Cluster {lab}",
                customdata=custom,
                marker=dict(size=7, opacity=0.9),
                hovertemplate="Gene: %{customdata}<br>Cluster: " + str(lab) + "<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Embedding dim 1",
        yaxis_title="Embedding dim 2",
        hovermode="closest",
        legend_title="Clusters",
    )
    return fig


def figure_to_html_with_click_highlight(
    fig: go.Figure,
    export: ExportHTML = ExportHTML(),
) -> str:
    """
    Convert a Plotly figure to HTML and inject JavaScript
    to enable cluster highlighting on click.

    Interactive behavior
    --------------------
    • clicking a point highlights its cluster
    • clicking the same cluster again resets
    • double-click resets the plot

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to export.

    export : ExportHTML
        HTML export configuration.

    Returns
    -------
    str
        HTML string containing the figure and JavaScript interaction code.
    """
    base_html = fig.to_html(include_plotlyjs=export.include_plotlyjs, full_html=export.full_html)

    import re
    m = re.search(r'<div id="([^"]+)"', base_html)
    if not m:
        raise RuntimeError("Could not locate Plotly div id in generated HTML.")
    div_id = m.group(1)

    js = f"""
<script>
document.addEventListener('DOMContentLoaded', function() {{
  var plot = document.getElementById('{div_id}');
  if (!plot) return;

  var lastCurve = null;

  function setAll(opacityArr) {{
    Plotly.restyle(plot, {{'marker.opacity': opacityArr}});
  }}

  function reset() {{
    var op = [];
    for (var i=0; i<plot.data.length; i++) op.push([0.9]);
    setAll(op);
    lastCurve = null;
  }}

  plot.on('plotly_click', function(evt) {{
    if (!evt || !evt.points || evt.points.length === 0) return;
    var curve = evt.points[0].curveNumber;

    if (lastCurve !== null && curve === lastCurve) {{
      reset();
      return;
    }}

    var op = [];
    for (var i=0; i<plot.data.length; i++) {{
      op.push([i === curve ? 0.95 : 0.12]);
    }}
    setAll(op);
    lastCurve = curve;
  }});

  plot.on('plotly_doubleclick', function() {{
    reset();
  }});
}});
</script>
"""

    if "</body>" in base_html:
        html = base_html.replace("</body>", js + "\n</body>")
    else:
        html = base_html + "\n" + js

    return html


def plot_embedding_click_highlight(
    affinity_matrix: np.ndarray,
    labels: np.ndarray,
    genes: Sequence[str],
    *,
    title: str = "Ensemble Embedding (click to highlight cluster)",
    embed: EmbedOptions = EmbedOptions(),
    save_html_to: Optional[PathLike] = None,
    export: ExportHTML = ExportHTML(),
    return_fig: bool = False,
    return_html: bool = False,
):
    """
    High-level helper to generate an embedding visualization.

    This function combines:

    1. spectral embedding
    2. Plotly figure creation
    3. optional HTML export

    Parameters
    ----------
    affinity_matrix : numpy.ndarray
        Matrix used for embedding.

    labels : numpy.ndarray
        Cluster labels.

    genes : Sequence[str]
        Gene identifiers.

    title : str
        Title of the plot.

    embed : EmbedOptions
        Embedding configuration.

    save_html_to : str or Path, optional
        File path where the HTML visualization should be saved.

    export : ExportHTML
        Export configuration.

    return_fig : bool
        If True, return the Plotly figure.

    return_html : bool
        If True, return the HTML string.

    Returns
    -------
    None | plotly.graph_objects.Figure | str | tuple
        Depending on the return flags.
    """
    fig = build_click_highlight_embedding_figure(
        affinity_matrix=affinity_matrix,
        labels=labels,
        genes=genes,
        title=title,
        embed=embed,
    )

    html = None
    if save_html_to is not None or return_html:
        html = figure_to_html_with_click_highlight(fig, export=export)
        if save_html_to is not None:
            p = _as_path(save_html_to)
            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(html, encoding="utf-8")
            _log_or_print(f"[ensemble] HTML saved at: {p}", export.verbose)

    if return_fig and return_html:
        return fig, html
    if return_fig:
        return fig
    if return_html:
        return html
    return None


# ──────────────────────────────────────────────────────────────
# Convenience pipelines
# ──────────────────────────────────────────────────────────────

def cspa_method(
    coincidence_matrix: np.ndarray,
    genes: Sequence[str],
    *,
    cspa: CSPAOptions,
    embed: EmbedOptions = EmbedOptions(),
    plot_title: str = "CSPA consensus embedding",
    save_html_to: Optional[PathLike] = None,
    export: ExportHTML = ExportHTML(),
    return_fig: bool = False,
    return_html: bool = False,
) -> Tuple[np.ndarray, Optional[go.Figure], Optional[str]]:
    """
    Run the full CSPA consensus clustering pipeline.

    This function performs the complete workflow:

    1. Apply spectral clustering to the coincidence matrix.
    2. Compute a spectral embedding for visualization.
    3. Generate an interactive Plotly visualization.
    4. Optionally export the visualization as HTML.

    Parameters
    ----------
    coincidence_matrix : numpy.ndarray
        Co-association matrix representing clustering similarity.

    genes : Sequence[str]
        Gene identifiers corresponding to each matrix row.

    cspa : CSPAOptions
        Configuration parameters for spectral clustering.

    embed : EmbedOptions
        Spectral embedding configuration.

    plot_title : str
        Title of the visualization.

    save_html_to : str or Path, optional
        Path where the HTML visualization will be written.

    export : ExportHTML
        HTML export configuration.

    return_fig : bool
        If True, return the Plotly figure.

    return_html : bool
        If True, return the HTML content.

    Returns
    -------
    tuple
        labels : numpy.ndarray
            Consensus cluster labels.

        fig : plotly.graph_objects.Figure or None
            Generated figure if requested.

        html : str or None
            HTML representation if requested.
    """
    labels = cspa_spectral_from_coincidence(coincidence_matrix, options=cspa)

    fig = None
    html = None
    if save_html_to is not None or return_fig or return_html:
        out = plot_embedding_click_highlight(
            affinity_matrix=coincidence_matrix,
            labels=labels,
            genes=genes,
            title=plot_title,
            embed=embed,
            save_html_to=save_html_to,
            export=export,
            return_fig=return_fig,
            return_html=return_html,
        )
        if return_fig and return_html:
            fig, html = out
        elif return_fig:
            fig = out
        elif return_html:
            html = out

    return labels, fig, html