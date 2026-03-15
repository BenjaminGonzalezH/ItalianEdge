"""
Ensemble clustering utilities (refactored).

Key features requested:
1) CSPA can receive the coincidence (co-association) matrix directly.
2) Keep SpectralEmbedding visualization (extra view vs Jaccard).
3) Interactive plots: clicking a point highlights its whole cluster and hover shows gene id.
4) Plurality Voting (PV) improved using *global stability* of partitions:
   - choose the most "central" partition (max mean similarity to others, default ARI)
   - align all partitions to that reference using Hungarian assignment
   - vote after alignment
5) MCLA is intentionally NOT implemented.

Design goals (consistent with the rest of your codebase):
- Robust validation
- Deterministic behavior (random_state)
- Optional HTML export (no residual files if used with tempfile.TemporaryDirectory in tests)
- Logging instead of prints

Author: ParetoInsight refactor (GPT)
"""

from __future__ import annotations

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
StabilityMetric = Literal["ari", "rand", "jaccard"]


# ──────────────────────────────────────────────────────────────
# Dataclasses (options)
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CSPAOptions:
    """Options for CSPA spectral clustering over a precomputed affinity matrix."""
    n_clusters: int
    assign_labels: Literal["kmeans", "discretize", "cluster_qr"] = "kmeans"
    random_state: int = 0


@dataclass(frozen=True)
class PVOptions:
    """
    Options for global-stability plurality voting.

    metric:
      - "ari": Adjusted Rand Index (recommended, stability-aware)
      - "rand": Rand Index (less informative when many clusters)
    """
    metric: StabilityMetric = "ari"


@dataclass(frozen=True)
class EmbedOptions:
    """Options for spectral embedding used in interactive plots."""
    n_components: int = 2
    random_state: int = 0


@dataclass(frozen=True)
class ExportHTML:
    """
    HTML export options for plotly figures.

    include_plotlyjs:
      - "cdn": smaller HTML, requires internet
      - "embed": standalone HTML (bigger)
      - True/False: plotly accepts bool too (True -> embed, False -> no js)
    """
    include_plotlyjs: Union[Literal["cdn", "embed"], bool] = "cdn"
    full_html: bool = True
    verbose: bool = False


# ──────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────

def _validate_square_numeric(matrix: np.ndarray, name: str) -> None:
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


def _validate_labels_matrix(labels_matrix: np.ndarray) -> None:
    if not isinstance(labels_matrix, np.ndarray):
        raise TypeError(f"labels_matrix must be numpy.ndarray, got {type(labels_matrix)}")
    if labels_matrix.ndim != 2:
        raise ValueError("labels_matrix must be 2D (m_partitions, n_elements).")
    if labels_matrix.shape[0] < 1 or labels_matrix.shape[1] < 1:
        raise ValueError("labels_matrix must be non-empty.")
    # allow ints/strings; we will factorize later


def _validate_genes(genes: Sequence[str], n: int) -> None:
    if genes is None:
        raise ValueError("genes must be provided (Sequence[str]).")
    if len(genes) != n:
        raise ValueError(f"Number of genes ({len(genes)}) does not match n_elements ({n}).")


def _validate_coincidence_matrix(coincidence: np.ndarray, *, tol_sym: float = 1e-8) -> None:
    """
    Coincidence (co-association) matrix expectations:
    - square, numeric
    - symmetric (within tolerance)
    - diagonal ~ 1
    - values typically in [0, 1] (we clip lightly if tiny numerical drift)
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
    Perform CSPA consensus clustering using SpectralClustering over a *coincidence* matrix.

    Parameters
    ----------
    coincidence_matrix:
        NxN co-association matrix, where entry (i,j) is the proportion of partitions
        where i and j are placed in the same cluster. Diagonal should be 1.
        This matrix is treated as an affinity matrix.
    options:
        CSPAOptions with n_clusters, assign_labels, random_state.

    Returns
    -------
    labels: np.ndarray shape (N,)
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


# ──────────────────────────────────────────────────────────────
# Global-stability Plurality Voting (PV)
# ──────────────────────────────────────────────────────────────

def _factorize_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Factorize arbitrary labels (ints/strings) into compact int codes 0..k-1.

    Returns:
        unique_labels, inverse_codes
    """
    unique, inv = np.unique(labels, return_inverse=True)
    return unique, inv.astype(np.int64, copy=False)


def _contingency_counts(inv_a: np.ndarray, inv_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Contingency counts between two factorized labelings a and b.

    Returns:
        nij: bincount over k_a * k_b (flattened)
        a_sum: counts per cluster in a
        b_sum: counts per cluster in b
        n: number of items
        k_b: number of clusters in b (needed for decoding)
    """
    if inv_a.shape != inv_b.shape:
        raise ValueError("Label vectors must have the same shape.")
    n = int(inv_a.size)
    if n == 0:
        raise ValueError("Label vectors must be non-empty.")

    k_a = int(inv_a.max()) + 1 if n else 0
    k_b = int(inv_b.max()) + 1 if n else 0

    codes = inv_a.astype(np.int64, copy=False) * k_b + inv_b.astype(np.int64, copy=False)
    nij = np.bincount(codes, minlength=k_a * k_b)

    a_sum = np.bincount(inv_a, minlength=k_a)
    b_sum = np.bincount(inv_b, minlength=k_b)

    return nij, a_sum, b_sum, n, k_b


def _comb2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    return (x * (x - 1)) // 2


def _adjusted_rand_from_counts(nij: np.ndarray, a_sum: np.ndarray, b_sum: np.ndarray, n: int) -> float:
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 0.0

    sum_nij = int(_comb2(nij).sum())
    sum_a = int(_comb2(a_sum).sum())
    sum_b = int(_comb2(b_sum).sum())

    expected = (sum_a * sum_b) / total_pairs
    max_index = 0.5 * (sum_a + sum_b)

    denom = max_index - expected
    if denom == 0:
        return 0.0
    return float((sum_nij - expected) / denom)


def _rand_from_counts(nij: np.ndarray, a_sum: np.ndarray, b_sum: np.ndarray, n: int) -> float:
    """
    Rand Index from contingency counts.

    Agreements = same-in-both + different-in-both
    same-in-both = sum_ij C(nij,2)
    different-in-both = total_pairs - same_in_a - same_in_b + same_in_both
    """
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 0.0

    same_in_both = int(_comb2(nij).sum())
    same_in_a = int(_comb2(a_sum).sum())
    same_in_b = int(_comb2(b_sum).sum())
    diff_in_both = total_pairs - same_in_a - same_in_b + same_in_both

    return float((same_in_both + diff_in_both) / total_pairs)


def _jaccard_from_counts(nij: np.ndarray, a_sum: np.ndarray, b_sum: np.ndarray, n: int) -> float:
    """
    Jaccard index for clustering partitions.

    J = a / (a + b + c)

    where:
      a = same cluster in both
      b = same cluster in A only
      c = same cluster in B only
    """

    same_in_both = int(_comb2(nij).sum())
    same_in_a = int(_comb2(a_sum).sum())
    same_in_b = int(_comb2(b_sum).sum())

    b = same_in_a - same_in_both
    c = same_in_b - same_in_both

    denom = same_in_both + b + c
    if denom == 0:
        return 0.0

    return float(same_in_both / denom)


def _similarity(labels_a: np.ndarray, labels_b: np.ndarray, metric: StabilityMetric) -> float:

    _, inv_a = _factorize_labels(labels_a)
    _, inv_b = _factorize_labels(labels_b)

    nij, a_sum, b_sum, n, _ = _contingency_counts(inv_a, inv_b)

    if metric == "ari":
        return _adjusted_rand_from_counts(nij, a_sum, b_sum, n)

    if metric == "rand":
        return _rand_from_counts(nij, a_sum, b_sum, n)

    if metric == "jaccard":
        return _jaccard_from_counts(nij, a_sum, b_sum, n)

    raise ValueError("metric must be 'ari', 'rand', or 'jaccard'")


def _best_reference_partition(labels_matrix: np.ndarray, metric: StabilityMetric) -> int:
    """
    Choose the most stable (central) partition by maximizing mean similarity to all others.
    """
    m = labels_matrix.shape[0]
    if m == 1:
        return 0

    sims = np.zeros(m, dtype=np.float64)
    for i in range(m):
        s = 0.0
        for j in range(m):
            if i == j:
                continue
            s += _similarity(labels_matrix[i], labels_matrix[j], metric=metric)
        sims[i] = s / (m - 1)
    return int(np.argmax(sims))


def _align_to_reference(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """
    Align 'current' labels to 'reference' labels using Hungarian assignment
    that maximizes overlaps.

    Returns aligned labels (same dtype as reference factorization -> int codes of reference clusters).
    """
    ref_unique, ref_inv = _factorize_labels(reference)
    cur_unique, cur_inv = _factorize_labels(current)

    k_ref = int(ref_inv.max()) + 1
    k_cur = int(cur_inv.max()) + 1

    # contingency matrix counts overlap between ref clusters (rows) and cur clusters (cols)
    codes = ref_inv * k_cur + cur_inv
    counts = np.bincount(codes, minlength=k_ref * k_cur).reshape(k_ref, k_cur)

    # Hungarian solves min-cost; use negative counts to maximize
    cost = -counts.astype(np.int64, copy=False)
    row_ind, col_ind = linear_sum_assignment(cost)

    # mapping from current cluster code -> reference cluster code
    mapping = {int(col_ind[r]): int(row_ind[r]) for r in range(len(row_ind))}

    aligned_codes = np.vectorize(lambda x: mapping.get(int(x), int(x)))(cur_inv)
    # return integer codes in reference space (0..k_ref-1 possibly with extras if unmatched)
    return aligned_codes.astype(np.int64, copy=False)


def plurality_voting_stable(
    labels_matrix: np.ndarray,
    options: PVOptions = PVOptions(),
) -> np.ndarray:
    """
    Global-stability plurality voting.

    Steps:
    1) pick most stable reference partition (max mean similarity to others)
    2) align all partitions to that reference using Hungarian mapping
    3) vote label per element using plurality

    Parameters
    ----------
    labels_matrix:
        shape (m_partitions, n_elements). Labels can be int or str.
    options:
        PVOptions(metric="ari" recommended)

    Returns
    -------
    consensus_labels: np.ndarray shape (n_elements,) of int codes (reference label space)
    """
    _validate_labels_matrix(labels_matrix)

    m, n = labels_matrix.shape
    if m == 1:
        # factorize to deterministic int codes
        _, inv = _factorize_labels(labels_matrix[0])
        return inv.astype(np.int64, copy=False)

    ref_idx = _best_reference_partition(labels_matrix, metric=options.metric)
    reference = labels_matrix[ref_idx]

    aligned = np.empty((m, n), dtype=np.int64)
    aligned[0] = _factorize_labels(reference)[1]  # ref as codes

    out_row = 1
    for r in range(m):
        if r == ref_idx:
            continue
        aligned[out_row] = _align_to_reference(reference, labels_matrix[r])
        out_row += 1

    # plurality vote per element
    consensus = np.zeros(n, dtype=np.int64)
    for i in range(n):
        vals, counts = np.unique(aligned[:, i], return_counts=True)
        consensus[i] = int(vals[int(np.argmax(counts))])

    return consensus


# ──────────────────────────────────────────────────────────────
# Interactive embedding plot (click-to-highlight cluster)
# ──────────────────────────────────────────────────────────────

def _spectral_embedding_2d(
    affinity_matrix: np.ndarray,
    embed: EmbedOptions,
) -> np.ndarray:
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
    Build a Plotly scatter figure where:
    - hover shows gene id and cluster label
    - clicking any point highlights the whole cluster (handled via JS in HTML export)

    Notes:
    - We build *one trace per cluster* (fast JS restyle: adjust trace opacity).
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
    Convert figure to HTML and inject JS so that clicking a point highlights its cluster trace.

    Behavior:
    - on click: selected trace opacity=1, others opacity=0.12
    - second click on same trace: reset
    - on double click: reset
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
    High-level helper: build figure + optionally export HTML + optionally return fig/html.

    Returns:
      - None by default
      - fig if return_fig=True
      - html if return_html=True
      - (fig, html) if both True
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

def ensemble_cspa(
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
    End-to-end CSPA:
      1) labels = CSPA spectral clustering from coincidence matrix
      2) interactive embedding plot with click-to-highlight
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


def ensemble_plurality_voting(
    labels_matrix: np.ndarray,
    coincidence_or_affinity_for_plot: np.ndarray,
    genes: Sequence[str],
    *,
    pv: PVOptions = PVOptions(),
    embed: EmbedOptions = EmbedOptions(),
    plot_title: str = "Plurality voting consensus embedding",
    save_html_to: Optional[PathLike] = None,
    export: ExportHTML = ExportHTML(),
    return_fig: bool = False,
    return_html: bool = False,
) -> Tuple[np.ndarray, Optional[go.Figure], Optional[str]]:
    """
    End-to-end PV (stable):
      1) consensus labels = plurality_voting_stable(labels_matrix)
      2) interactive embedding plot over provided affinity matrix
         (typically the coincidence matrix used also by CSPA)

    Note:
      PV itself does NOT require the affinity matrix; it is only used for embedding visualization.
    """
    _validate_labels_matrix(labels_matrix)
    _validate_square_numeric(coincidence_or_affinity_for_plot, "coincidence_or_affinity_for_plot")

    n = coincidence_or_affinity_for_plot.shape[0]
    if labels_matrix.shape[1] != n:
        raise ValueError(
            f"labels_matrix has n_elements={labels_matrix.shape[1]} but affinity plot matrix has n={n}"
        )
    _validate_genes(genes, n)

    consensus = plurality_voting_stable(labels_matrix, options=pv)

    fig = None
    html = None
    if save_html_to is not None or return_fig or return_html:
        out = plot_embedding_click_highlight(
            affinity_matrix=coincidence_or_affinity_for_plot,
            labels=consensus,
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

    return consensus, fig, html
