"""
Heatmaps utilities.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass                   # Decorator to automatically generate special methods (e.g., __init__).
from pathlib import Path                            # Object-oriented filesystem path handling.
from typing import Optional, Literal, Union, Tuple  # Improve type hints and function signatures.
import numpy as np                                  # Efficient numerical computations.
import plotly.graph_objects as go                   # Plotting graphs.
import logging                                      # Advanced logging system for status and error messages.

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)                # Initialize module-level logger.

# ──────────────────────────────────────────────────────────────────────────────
# Data Types
# ──────────────────────────────────────────────────────────────────────────────
PathLike = Union[str, Path]
DownsampleMode = Literal["none", "pool_mean", "pool_max"]


# ──────────────────────────────────────────────────────────────────────────────
# Classes
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HeatmapExportOptions:
    """
    Export options for saving an HTML plot.

    Attributes:
        include_plotlyjs: "cdn" keeps HTML smaller; "embed" is standalone but heavier.
        full_html: If True, writes a full HTML document; else a div snippet.
        verbose: If True, prints extra status messages (still logs always).
    """
    include_plotlyjs: Union[Literal["cdn", "embed"], bool] = "cdn"
    full_html: bool = False
    verbose: bool = True


@dataclass(frozen=True)
class HeatmapScaleOptions:
    """
    Large-scale options to keep plots usable for big N.

    Attributes:
        max_dim: If matrix is larger than this in any dimension, it will be downsampled.
        downsample_mode: pooling strategy ("pool_mean" recommended).
        webgl_threshold: If n*m >= threshold, prefer WebGL heatmap (Heatmapgl) when possible.
        force_webgl: Always use Heatmapgl (can help huge matrices but has feature differences).
    """
    max_dim: int = 1200
    downsample_mode: DownsampleMode = "pool_mean"

# ──────────────────────────────────────────────────────────────────────────────
# Internal functions
# ──────────────────────────────────────────────────────────────────────────────

def _log_or_print(msg: str, verbose: bool) -> None:
    """
    Library-friendly output handler:
    - Always logs the message using the module logger.
    - Optionally prints the message if verbose=True.
    """
    logger.info(msg)
    if verbose:
        print(msg)


def _as_path(p: PathLike) -> Path:
    """Ensure the input is converted to a Path object."""
    return p if isinstance(p, Path) else Path(p)

def _validate_matrix_2d(matrix: np.ndarray, name: str = "matrix") -> None:
    """Checks the status of the input matrix or matrices"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got: {type(matrix)}")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D, got ndim={matrix.ndim}")
    if matrix.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"{name} must be numeric dtype, got: {matrix.dtype}")

def _pool_downsample(
    matrix: np.ndarray,
    max_dim: int,
    mode: DownsampleMode
) -> Tuple[np.ndarray, int, int]:
    """
    Downsample by block pooling to keep plots responsive.

    Returns:
        downsampled_matrix, pool_h, pool_w
    """
    if mode == "none":
        return matrix, 1, 1

    h, w = matrix.shape
    if h <= max_dim and w <= max_dim:
        return matrix, 1, 1

    # Compute pooling factors
    pool_h = int(np.ceil(h / max_dim))
    pool_w = int(np.ceil(w / max_dim))

    # Pad to multiples of pooling sizes (pad with NaN to ignore in pooling)
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


def _choose_heatmap_trace(
    z: np.ndarray,
    colorscale: str,
    z_label: str,
    hovertemplate: Optional[str],
    scale_opts: HeatmapScaleOptions,
    showscale: bool = True,
    colorbar: Optional[dict] = None,
) -> go.Heatmap:
    """
    Select Heatmap vs Heatmapgl based on size.
    """
    heatmap_cls = go.Heatmap

    cb = colorbar if colorbar is not None else dict(title=z_label)

    trace = heatmap_cls(
        z=z,
        colorscale=colorscale,
        showscale=showscale,
        colorbar=cb,
    )

    if hovertemplate is not None:
        trace.update(hovertemplate=hovertemplate)

    return trace

def _to_html(fig: go.Figure, options: HeatmapExportOptions) -> str:
    """If it is stablish by user, the plot is completely HTML"""
    return fig.to_html(include_plotlyjs=options.include_plotlyjs, full_html=options.full_html)

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

def plot_html_heatmap(
    matrix: np.ndarray,
    save_filepath: Optional[PathLike] = "heatmap.html",
    x_label: str = "",
    y_label: str = "",
    title: str = "Heatmap",
    colorscale: str = "Viridis",
    z_label: str = "Valor",
    tooltip_format: Optional[str] = "X: %{x}<br>Y: %{y}<br>Valor: %{z:.4g}<extra></extra>",
    export: HeatmapExportOptions = HeatmapExportOptions(),
    scale: HeatmapScaleOptions = HeatmapScaleOptions(),
    return_fig: bool = False,
    return_html: bool = False,
):
    """
    Create an interactive heatmap and optionally save as HTML.

    Large-scale behavior:
    - If matrix > scale.max_dim, it is downsampled by pooling.

    Returns:
        None by default.
        If return_fig=True -> returns plotly Figure.
        If return_html=True -> returns HTML string.
        If both True -> returns (Figure, HTML).
    """
    _validate_matrix_2d(matrix, "matrix")

    z, pool_h, pool_w = _pool_downsample(matrix, scale.max_dim, scale.downsample_mode)

    if pool_h > 1 or pool_w > 1:
        _log_or_print(
            f"[heatmap] Downsampled from {matrix.shape} to {z.shape} "
            f"(pool_h={pool_h}, pool_w={pool_w}, mode={scale.downsample_mode}).",
            export.verbose,
        )

    fig = go.Figure()

    trace = _choose_heatmap_trace(
        z=z,
        colorscale=colorscale,
        z_label=z_label,
        hovertemplate=tooltip_format,
        scale_opts=scale,
        showscale=True,
        colorbar=dict(title=z_label),
    )
    fig.add_trace(trace)

    subtitle = ""
    if pool_h > 1 or pool_w > 1:
        subtitle = f" (downsampled: {matrix.shape} → {z.shape})"

    fig.update_layout(
        title=title + subtitle,
        xaxis_title=x_label,
        yaxis_title=y_label,
        autosize=True,
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(constrain="domain"),
        hovermode="closest",
    )

    html: Optional[str] = None
    if save_filepath is not None or return_html:
        html = _to_html(fig, export)
        if save_filepath is not None:
            out = _write_text(save_filepath, html)
            _log_or_print(f"[heatmap] HTML saved at: {out}", export.verbose)

    if return_fig and return_html:
        return fig, html
    if return_fig:
        return fig
    if return_html:
        return html
    return None


def plot_dual_heatmap_two_colors(
    matrix_upper: np.ndarray,
    matrix_lower: np.ndarray,
    save_filepath: Optional[PathLike] = "dual_heatmap.html",
    x_label: str = "Solutions",
    y_label: str = "Solutions",
    title: str = "Jaccard vs Wang",
    colorscale_upper: str = "Viridis",
    colorscale_lower: str = "Plasma",
    export: HeatmapExportOptions = HeatmapExportOptions(),
    scale: HeatmapScaleOptions = HeatmapScaleOptions(),
    return_fig: bool = False,
    return_html: bool = False,
):
    """
    Dual heatmap: upper triangle uses one colorscale, lower triangle another.
    Includes symmetric highlighting using JS, without re-reading the saved file.

    Large-scale behavior:
    - Optional downsampling (pooling) to keep plot responsive.
    - Avoids building n×n hovertext matrices (RAM heavy).
    """
    _validate_matrix_2d(matrix_upper, "matrix_upper")
    _validate_matrix_2d(matrix_lower, "matrix_lower")

    if matrix_upper.shape != matrix_lower.shape:
        raise ValueError("matrix_upper and matrix_lower must have the same shape.")
    if matrix_upper.shape[0] != matrix_upper.shape[1]:
        raise ValueError("Dual triangular heatmap requires square matrices.")

    # Downsample both consistently
    z_u, pool_h, pool_w = _pool_downsample(matrix_upper, scale.max_dim, scale.downsample_mode)
    z_l, pool_h2, pool_w2 = _pool_downsample(matrix_lower, scale.max_dim, scale.downsample_mode)
    if (pool_h, pool_w) != (pool_h2, pool_w2):
        # This should never happen with same shape, but keep it explicit.
        raise RuntimeError("Internal error: inconsistent downsampling factors.")

    if pool_h > 1 or pool_w > 1:
        _log_or_print(
            f"[dual_heatmap] Downsampled from {matrix_upper.shape} to {z_u.shape} "
            f"(pool_h={pool_h}, pool_w={pool_w}, mode={scale.downsample_mode}).",
            export.verbose,
        )

    n = z_u.shape[0]
    # Masks (vectorized, no Python nested loops)
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)

    # Put NaN where not used so Plotly doesn't draw those cells
    z_upper = np.where(upper_mask, z_u, np.nan)
    z_lower = np.where(lower_mask, z_l, np.nan)

    fig = go.Figure()

    # Hover: keep it light. Avoid customdata hover matrices.
    # We show coords + value; plotly already provides %{x}, %{y}, %{z}.
    hover_upper = "i: %{y}<br>j: %{x}<br>Jaccard: %{z:.4g}<extra></extra>"
    hover_lower = "i: %{y}<br>j: %{x}<br>Wang: %{z:.4g}<extra></extra>"

    fig.add_trace(
        _choose_heatmap_trace(
            z=z_upper,
            colorscale=colorscale_upper,
            z_label="Jaccard",
            hovertemplate=hover_upper,
            scale_opts=scale,
            showscale=True,
            colorbar=dict(title="Jaccard", x=1.0, y=0.75, len=0.4),
        )
    )

    fig.add_trace(
        _choose_heatmap_trace(
            z=z_lower,
            colorscale=colorscale_lower,
            z_label="Wang",
            hovertemplate=hover_lower,
            scale_opts=scale,
            showscale=True,
            colorbar=dict(title="Wang", x=1.0, y=0.25, len=0.4),
        )
    )

    # Highlight traces: scatter squares
    for _ in range(2):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                symbol="square",
                size=20,
                color="rgba(255, 0, 0, 0.0)",
                line=dict(color="red", width=2),
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

    subtitle = ""
    if pool_h > 1 or pool_w > 1:
        subtitle = f" (downsampled: {matrix_upper.shape} → {z_u.shape})"

    fig.update_layout(
        title=title + subtitle,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(constrain="domain"),
        hovermode="closest",
        annotations=[
            dict(
                text="Pasa el cursor sobre una celda para ver su coordenada simétrica",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.15
            )
        ],
    )

    # Build HTML once, inject JS directly (no read/modify/write cycle)
    html: Optional[str] = None
    if save_filepath is not None or return_html:
        base_html = _to_html(fig, export)

        # We need the plot div id. Plotly emits one in the HTML.
        # Use a robust pattern: first occurrence of <div id="...">
        import re
        m = re.search(r'<div id="([^"]+)"', base_html)
        if not m:
            raise RuntimeError("Could not locate Plotly div id in generated HTML.")
        div_id = m.group(1)

        js_code = f"""
<script>
document.addEventListener('DOMContentLoaded', function() {{
  var myPlot = document.getElementById('{div_id}');
  if (!myPlot) return;

  myPlot.on('plotly_hover', function(data) {{
    if (!data || !data.points || data.points.length === 0) return;
    var p = data.points[0];
    var xCoord = p.x;
    var yCoord = p.y;

    // Highlight hovered cell (trace index 2)
    Plotly.restyle(myPlot, {{ x: [[xCoord]], y: [[yCoord]] }}, [2]);

    // Highlight symmetric cell (trace index 3)
    Plotly.restyle(myPlot, {{ x: [[yCoord]], y: [[xCoord]] }}, [3]);
  }});

  myPlot.on('plotly_unhover', function() {{
    Plotly.restyle(myPlot, {{ x: [[null]], y: [[null]] }}, [2, 3]);
  }});
}});
</script>
"""

        # Inject before </body> if present; else append.
        if "</body>" in base_html:
            html = base_html.replace("</body>", js_code + "\n</body>")
        else:
            html = base_html + "\n" + js_code

        if save_filepath is not None:
            out = _write_text(save_filepath, html)
            _log_or_print(f"[dual_heatmap] HTML saved at: {out}", export.verbose)

    if return_fig and return_html:
        return fig, html
    if return_fig:
        return fig
    if return_html:
        return html
    return None
