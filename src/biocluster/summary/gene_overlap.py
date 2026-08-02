"""
gene_overlap_summary: Merges and generalizes the former ``differences_summary.py`` (disjoint /
symmetric-difference genes) and ``similarities_summary.py`` (shared / intersection genes) modules
into a single, mode-parameterized API.

Functions
1. compute_gene_overlap_dataframe -
   Generates a new DataFrame including shared ("intersection") or disjoint
   ("symmetric difference") gene sets for EVERY cluster pair in the input
   DataFrame — no similarity/threshold-based filtering is applied, since all
   solutions in the set are considered.

2. compute_gene_frequencies -
   Count how often each gene appears across the shared/disjoint gene sets.

3. compute_frequency_cutoff -
   Automatically determine the frequency value that best separates
   "incidental" genes (appearing in few cluster pairs) from "recurrent"
   genes (appearing consistently), using the knee/elbow of the
   rank-frequency curve. Replaces a manually-chosen ``min_gene_frequency``.

4. plot_frequency_cutoff -
   Histogram of gene frequencies with the cutoff from (3) highlighted —
   meant to be called with the same ``freq_df`` and cutoff value used to
   drive ``summarize_genes``, so the cut is visually auditable.

5. summarize_genes -
   Summarize gene frequencies, the biological (gene-gene) similarity
   submatrix, and the set co-occurrence Jaccard matrix, restricted to genes
   at or above a frequency cutoff (auto-computed via (3) if not given
   explicitly). Returns the selected gene list and the similarity submatrix
   directly, alongside the supporting frequency/co-occurrence data.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
import warnings
from collections import Counter
from collections.abc import Sequence
from typing import Literal, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

OverlapMode = Literal["shared", "disjoint"]
CutoffMethod = Literal["knee"]

# Nested-sequence mapping: solution_cluster_matrix[solution_idx][cluster_idx] -> set of genes.
# This is the same structure produced by ``solution_cluster_matrix()``
# (solutioncluster_matrix.py), i.e. List[List[Set[str]]] indexed by two
# integers — NOT a Dict[Tuple[str, str], Set[str]] as the original
# docstrings suggested (that example didn't match how the code actually
# indexed it).
SolutionClusterMatrix = Sequence[Sequence[set[str]]]


# ──────────────────────────────────────────────────────────────────────────────
# Core function: shared / disjoint genes for every cluster pair
# ──────────────────────────────────────────────────────────────────────────────


def compute_gene_overlap_dataframe(
    df: pd.DataFrame,
    solution_cluster_matrix: SolutionClusterMatrix,
    mode: OverlapMode = "shared",
) -> pd.DataFrame:
    """
    Compute shared or disjoint genes for EVERY cluster pair in ``df``.

    Unlike the original ``compute_shared_genes_dataframe`` /
    ``compute_disjoint_genes_dataframe``, this function does not filter rows
    by a similarity/threshold column: every solution pair in the input
    DataFrame is considered, since the full set of solutions is meant to be
    analyzed.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing cluster pair relationships. Must include:
        "Solution 1", "Solution 2", "Cluster 1", "Cluster 2".
    solution_cluster_matrix : Sequence[Sequence[Set[str]]]
        solution_cluster_matrix[s][c] -> set of genes in cluster c of
        solution s (the output of ``solution_cluster_matrix()``).
    mode : {"shared", "disjoint"}
        "shared"   -> intersection of the two clusters' gene sets.
        "disjoint" -> symmetric difference of the two clusters' gene sets.

    Returns
    -------
    pd.DataFrame
        Original pair identifiers plus a "Shared Genes" or "Disjoint Genes"
        column (as Python sets), depending on ``mode``.
    """
    required_columns = {"Solution 1", "Solution 2", "Cluster 1", "Cluster 2"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if mode not in ("shared", "disjoint"):
        raise ValueError("mode must be 'shared' or 'disjoint'.")

    gene_column = "Shared Genes" if mode == "shared" else "Disjoint Genes"

    overlap_list = []
    for _, row in df.iterrows():
        genes_1 = solution_cluster_matrix[int(row["Solution 1"])][int(row["Cluster 1"])]
        genes_2 = solution_cluster_matrix[int(row["Solution 2"])][int(row["Cluster 2"])]

        overlap = (
            genes_1 & genes_2
            if mode == "shared"
            else (genes_1 - genes_2) | (genes_2 - genes_1)
        )

        overlap_list.append(overlap)

    output_df = df[["Solution 1", "Cluster 1", "Solution 2", "Cluster 2"]].copy()
    output_df[gene_column] = overlap_list

    return output_df


# ─────────────────────────────────────────────────────────────────────────────
# Frequency counting
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_gene_sets(df: pd.DataFrame, gene_column: str) -> list[list[str]]:
    """Normalize gene collections into clean lists of strings."""
    normalized = []
    for genes in df[gene_column]:
        if isinstance(genes, (set, list, tuple)):
            cleaned = [str(g).strip() for g in genes if pd.notna(g)]
            cleaned = [g for g in cleaned if g]
            if cleaned:
                normalized.append(cleaned)
    return normalized


def compute_gene_frequencies(df: pd.DataFrame, gene_column: str) -> pd.DataFrame:
    """
    Count how often each gene appears across the gene sets in ``gene_column``
    (e.g. the "Shared Genes" or "Disjoint Genes" column produced by
    ``compute_gene_overlap_dataframe``).

    Returns
    -------
    pd.DataFrame
        Columns: "Gene", "Frequency", sorted descending by frequency.
    """
    gene_sets = _normalize_gene_sets(df, gene_column)
    all_genes = [g for genes in gene_sets for g in genes]

    if not all_genes:
        return pd.DataFrame(columns=["Gene", "Frequency"])

    counter = Counter(all_genes)
    return (
        pd.DataFrame([{"Gene": g, "Frequency": c} for g, c in counter.items()])
        .sort_values(["Frequency", "Gene"], ascending=[False, True])
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Automatic frequency cutoff + companion histogram
# ─────────────────────────────────────────────────────────────────────────────


def compute_frequency_cutoff(
    freq_df: pd.DataFrame,
    method: CutoffMethod = "knee",
) -> int:
    """
    Automatically determine the frequency cutoff that best separates
    "incidental" genes (appearing in few cluster pairs) from "recurrent"
    genes (appearing consistently), using the knee/elbow of the
    rank-frequency curve.

    Genes are sorted by descending frequency, producing a decreasing,
    typically L-shaped curve: a short, steep head of a few high-frequency
    "recurrent" genes, followed by a long, flat tail of low-frequency
    "incidental" ones. The knee is located as the point of maximum
    perpendicular distance from the straight chord joining the first and
    last points of that curve (both axes normalized to [0, 1]) — i.e. the
    corner where the steep head gives way to the flat tail. The knee marks
    the onset of the incidental tail, so the cutoff returned is the lowest
    frequency among the recurrent (pre-knee) genes.

    This requires no manually-chosen threshold and, unlike a
    bimodal-histogram method (e.g. Otsu), is well suited to the strongly
    skewed, long-tailed distributions typical of gene frequencies, where
    most genes appear only once or twice.

    Parameters
    ----------
    freq_df : pd.DataFrame
        Output of ``compute_gene_frequencies`` (must contain a "Frequency"
        column).
    method : {"knee"}
        Cutoff-detection method. Only "knee" (the rank-frequency
        knee/elbow) is implemented; kept as a parameter for future
        extension.

    Returns
    -------
    int
        The cutoff frequency value. Genes with Frequency >= cutoff are
        considered "recurrent" and should be kept; genes below it are
        considered incidental. Degenerate inputs (empty, all-equal, or
        all-singleton frequencies, or fewer than three genes) yield no
        separation and return the minimum frequency, keeping every gene.
    """
    if method != "knee":
        raise ValueError("Only method='knee' is currently implemented.")

    if freq_df.empty:
        return 0

    frequencies = freq_df["Frequency"].to_numpy().astype(int)
    max_freq = int(frequencies.max())
    min_freq = int(frequencies.min())

    # No meaningful elbow (no spread, or too few points): keep every gene.
    if max_freq <= 1 or max_freq == min_freq or frequencies.size < 3:
        return min_freq

    # Rank-frequency curve: frequencies sorted in descending order.
    freqs_desc = np.sort(frequencies)[::-1].astype(np.float64)
    n = freqs_desc.size

    # Normalize both axes to [0, 1] so rank and frequency are comparable.
    x = np.arange(n, dtype=np.float64) / (n - 1)
    y = (freqs_desc - min_freq) / (max_freq - min_freq)

    # Perpendicular distance from each point to the chord joining
    # (x=0, y=1) and (x=1, y=0) is proportional to |1 - x - y|. Its maximum
    # is the knee: the corner where the steep head meets the flat tail.
    distance = np.abs(1.0 - x - y)
    knee_idx = int(np.argmax(distance))

    # Guard the degenerate case where the maximum lands on the first point.
    if knee_idx == 0:
        return min_freq

    # Genes before the knee are the recurrent ones; the cutoff is the
    # lowest frequency among them, so that Frequency >= cutoff keeps exactly
    # the head of the curve (ties at the cutoff frequency are kept together).
    return int(freqs_desc[knee_idx - 1])


def plot_frequency_cutoff(
    freq_df: pd.DataFrame,
    cutoff: int,
    title: str = "Gene Frequency Cutoff",
) -> go.Figure:
    """
    Rank-frequency curve (genes ordenados por frecuencia descendente),
    igual a la curva que usa compute_frequency_cutoff para hallar el codo.
    """
    if freq_df.empty:
        return go.Figure()

    frequencies = freq_df["Frequency"].to_numpy().astype(int)
    freqs_desc = np.sort(frequencies)[::-1]
    ranks = np.arange(1, freqs_desc.size + 1)

    colors = ["#1f77b4" if f >= cutoff else "#b0b0b0" for f in freqs_desc]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=ranks,
                y=freqs_desc,
                mode="lines+markers",
                marker={"color": colors, "size": 6},
                line={"color": "#c0c0c0", "width": 1},
                hovertemplate="Rank: %{x}<br>Frequency: %{y}<extra></extra>",
            )
        ]
    )

    # Posición del codo: último rank cuya frecuencia es >= cutoff
    knee_rank = int(np.sum(freqs_desc >= cutoff))

    fig.add_vline(
        x=knee_rank + 0.5,
        line_dash="dash",
        line_color="red",
        annotation_text=f"cutoff = {cutoff}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Gene rank (sorted by descending frequency)",
        yaxis_title="Frequency (# of cluster pairs)",
        template="plotly_white",
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Biological submatrix + co-occurrence
# ─────────────────────────────────────────────────────────────────────────────


def _build_frequency_plot(
    df: pd.DataFrame,
    top_n: Optional[int],
    label: str,
    cutoff: Optional[int] = None,
) -> go.Figure:
    """
    Build a horizontal lollipop plot of the top-N genes by frequency,
    highlighting which fall above/below the recurrent-gene cutoff from
    compute_frequency_cutoff.
    """
    plot_df = df.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    # Orden ascendente para que el gen de mayor frecuencia quede arriba
    plot_df = plot_df.sort_values("Frequency", ascending=True)

    colors: Union[str, list[str]]
    if cutoff is not None:
        colors = ["#1f77b4" if f >= cutoff else "#b0b0b0" for f in plot_df["Frequency"]]
    else:
        colors = "#1f77b4"

    fig = go.Figure()

    # Líneas del lollipop (una por gen)
    for _, row in plot_df.iterrows():
        fig.add_shape(
            type="line",
            x0=0,
            x1=row["Frequency"],
            y0=row["Gene"],
            y1=row["Gene"],
            line={"color": "#d8d8d8", "width": 2},
            layer="below",
        )

    # Puntos
    fig.add_trace(
        go.Scatter(
            x=plot_df["Frequency"],
            y=plot_df["Gene"],
            mode="markers+text",
            marker={
                "color": colors,
                "size": 11,
                "line": {"color": "white", "width": 1},
            },
            text=plot_df["Frequency"],
            textposition="middle right",
            hovertemplate="Gene: %{y}<br>Frequency: %{x}<extra></extra>",
            showlegend=False,
        )
    )

    if cutoff is not None:
        fig.add_vline(
            x=cutoff,
            line_dash="dash",
            line_color="red",
            annotation_text=f"cutoff = {cutoff}",
            annotation_position="top",
        )

    fig.update_layout(
        title=f"Top {len(plot_df)} Genes by Frequency in {label}",
        xaxis_title="Frequency (# of cluster pairs)",
        yaxis_title="Gene",
        template="plotly_white",
        height=max(350, 32 * len(plot_df)),  # espacio cómodo por gen
        margin={"l": 120, "r": 60, "t": 60, "b": 40},
        xaxis={
            "range": [0, plot_df["Frequency"].max() * 1.15]
        },  # espacio para las etiquetas
    )
    return fig


def _extract_biological_submatrix(
    selected_genes: list[str],
    gene_ids: list[str],
    matrix: np.ndarray,
) -> pd.DataFrame:
    """Extract biological similarity submatrix."""
    gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
    present = [g for g in selected_genes if g in gene_to_idx]
    missing = set(selected_genes) - set(present)

    if missing:
        warnings.warn(
            f"{len(missing)} genes not found in similarity matrix.", stacklevel=2
        )

    if not present:
        return pd.DataFrame()

    idx = [gene_to_idx[g] for g in present]
    sub = matrix[np.ix_(idx, idx)]
    return pd.DataFrame(sub, index=present, columns=present)


def _compute_cooccurrence_jaccard(
    gene_sets: list[list[str]],
    selected_genes: list[str],
) -> pd.DataFrame:
    """
    Compute co-occurrence Jaccard matrix using vectorized linear algebra.

    C = X.T @ X
    J = C / (f_i + f_j - C)
    """
    if not selected_genes:
        return pd.DataFrame()

    gene_to_idx = {g: i for i, g in enumerate(selected_genes)}
    n_sets = len(gene_sets)
    n_genes = len(selected_genes)

    X = np.zeros((n_sets, n_genes), dtype=np.uint64)
    for i, genes in enumerate(gene_sets):
        for g in set(genes):
            if g in gene_to_idx:
                X[i, gene_to_idx[g]] = 1

    C = X.T @ X
    diag = np.diag(C)
    denom = diag[:, None] + diag[None, :] - C

    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(denom > 0, C / denom, 0.0)

    return pd.DataFrame(J, index=selected_genes, columns=selected_genes)


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────


class GeneSummaryResult(NamedTuple):
    """Result of ``summarize_genes``."""

    selected_genes: list[str]
    similarity_submatrix: pd.DataFrame
    frequency_df: pd.DataFrame
    frequency_figure: go.Figure
    cooccurrence_df: pd.DataFrame
    cutoff_used: int


def summarize_genes(
    df: pd.DataFrame,
    gene_column: str,
    gene_ids: Sequence[str],
    gene_similarity_matrix: np.ndarray,
    *,
    min_gene_frequency: Optional[int] = None,
    top_n_genes_for_plot: Optional[int] = 30,
    max_genes: Optional[int] = None,
    label: str = "Genes",
) -> GeneSummaryResult:
    """
    Summarize genes appearing in shared/disjoint gene sets, restricted to
    genes at or above a frequency cutoff.

    Generalizes the former ``summarize_shared_genes`` / ``summarize_disjoint_genes``:
    works on any gene-set column ("Shared Genes", "Disjoint Genes", or a
    custom name), controlled via ``gene_column``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``compute_gene_overlap_dataframe`` (or any DataFrame with
        a column of gene sets/lists).
    gene_column : str
        Name of the column containing gene sets/lists (e.g. "Shared Genes").
    gene_ids : Sequence[str]
        Ordered gene identifiers matching ``gene_similarity_matrix``'s rows/cols.
    gene_similarity_matrix : np.ndarray
        Square gene-gene similarity matrix.
    min_gene_frequency : int, optional
        Minimum frequency for a gene to be kept. If None (default), it is
        determined automatically via ``compute_frequency_cutoff``
        (knee/elbow method) — this is the recommended usage, replacing a
        manually guessed threshold.
    top_n_genes_for_plot : int, optional
        Limit the frequency bar plot to the top N selected genes.
    max_genes : int, optional
        Hard cap on the number of selected genes (after frequency filtering).
    label : str
        Human-readable label used in plot titles (e.g. "Shared Genes").

    Returns
    -------
    GeneSummaryResult
        NamedTuple with:
        - selected_genes: genes kept after the frequency cutoff.
        - similarity_submatrix: gene-gene similarity submatrix for those genes.
        - frequency_df: full gene frequency table (before cutoff filtering).
        - frequency_figure: bar plot of frequencies for the selected genes.
        - cooccurrence_df: co-occurrence Jaccard matrix for the selected genes.
        - cutoff_used: the frequency cutoff actually applied (useful to feed
          into ``plot_frequency_cutoff`` for the audit histogram).
    """
    if gene_column not in df.columns:
        raise ValueError(f"Missing column: {gene_column}")

    gene_ids = list(map(str, gene_ids))

    if gene_similarity_matrix.ndim != 2:
        raise ValueError("gene_similarity_matrix must be 2D")
    if gene_similarity_matrix.shape[0] != gene_similarity_matrix.shape[1]:
        raise ValueError("gene_similarity_matrix must be square")
    if gene_similarity_matrix.shape[0] != len(gene_ids):
        raise ValueError("Matrix size must match gene_ids length")

    empty_result = GeneSummaryResult(
        selected_genes=[],
        similarity_submatrix=pd.DataFrame(),
        frequency_df=pd.DataFrame(columns=["Gene", "Frequency"]),
        frequency_figure=go.Figure(),
        cooccurrence_df=pd.DataFrame(),
        cutoff_used=0,
    )

    gene_sets = _normalize_gene_sets(df, gene_column)
    if not gene_sets:
        return empty_result

    # ── Frequency + automatic cutoff ────────────────────────────────────────
    freq_df = compute_gene_frequencies(df, gene_column)

    cutoff = (
        compute_frequency_cutoff(freq_df)
        if min_gene_frequency is None
        else min_gene_frequency
    )

    selected_genes = freq_df.loc[freq_df["Frequency"] >= cutoff, "Gene"].tolist()
    if max_genes is not None:
        selected_genes = selected_genes[:max_genes]

    if not selected_genes:
        return empty_result._replace(frequency_df=freq_df, cutoff_used=cutoff)

    # ── Frequency plot (selected genes only) ────────────────────────────────
    fig = _build_frequency_plot(
        freq_df[freq_df["Gene"].isin(selected_genes)],
        top_n_genes_for_plot,
        label,
    )

    # ── Biological submatrix ─────────────────────────────────────────────────
    bio_df = _extract_biological_submatrix(
        selected_genes, gene_ids, gene_similarity_matrix
    )

    # ── Co-occurrence (vectorized) ───────────────────────────────────────────
    cooc_df = _compute_cooccurrence_jaccard(gene_sets, selected_genes)

    return GeneSummaryResult(
        selected_genes=selected_genes,
        similarity_submatrix=bio_df,
        frequency_df=freq_df,
        frequency_figure=fig,
        cooccurrence_df=cooc_df,
        cutoff_used=cutoff,
    )
