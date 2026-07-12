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
   genes (appearing consistently), using Otsu's method applied to the
   histogram of gene frequencies. Replaces a manually-chosen
   ``min_gene_frequency``.

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
from typing      import List, Literal, NamedTuple, Optional, Sequence, Set
from collections import Counter
import numpy                as np
import pandas               as pd
import plotly.graph_objects as go
import warnings

OverlapMode = Literal["shared", "disjoint"]
CutoffMethod = Literal["otsu"]

# Nested-sequence mapping: solution_cluster_matrix[solution_idx][cluster_idx] -> set of genes.
# This is the same structure produced by ``solution_cluster_matrix()``
# (solutioncluster_matrix.py), i.e. List[List[Set[str]]] indexed by two
# integers — NOT a Dict[Tuple[str, str], Set[str]] as the original
# docstrings suggested (that example didn't match how the code actually
# indexed it).
SolutionClusterMatrix = Sequence[Sequence[Set[str]]]


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

        if mode == "shared":
            overlap = genes_1 & genes_2
        else:
            overlap = (genes_1 - genes_2) | (genes_2 - genes_1)

        overlap_list.append(overlap)

    output_df = df[["Solution 1", "Cluster 1", "Solution 2", "Cluster 2"]].copy()
    output_df[gene_column] = overlap_list

    return output_df


# ─────────────────────────────────────────────────────────────────────────────
# Frequency counting
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_gene_sets(df: pd.DataFrame, gene_column: str) -> List[List[str]]:
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
    method: CutoffMethod = "otsu",
) -> int:
    """
    Automatically determine the frequency cutoff that best separates
    "incidental" genes (appearing in few cluster pairs) from "recurrent"
    genes (appearing consistently), using Otsu's method on the histogram
    of gene frequencies.

    Otsu's method scans every possible split point of the histogram and
    picks the one that maximizes the between-class variance (i.e. the
    split that makes the "low frequency" and "high frequency" groups as
    internally homogeneous, and as different from each other, as possible).
    It requires no manually-chosen threshold.

    Parameters
    ----------
    freq_df : pd.DataFrame
        Output of ``compute_gene_frequencies`` (must contain a "Frequency"
        column).
    method : {"otsu"}
        Cutoff-detection method. Only "otsu" is implemented; kept as a
        parameter for future extension (e.g. a knee/elbow-based method).

    Returns
    -------
    int
        The cutoff frequency value. Genes with Frequency >= cutoff are
        considered "recurrent" and should be kept; genes below it are
        considered incidental.
    """
    if method != "otsu":
        raise ValueError("Only method='otsu' is currently implemented.")

    if freq_df.empty:
        return 0

    frequencies = freq_df["Frequency"].to_numpy().astype(int)
    max_freq = int(frequencies.max())

    if max_freq <= 1:
        return int(frequencies.min())

    hist = np.bincount(frequencies, minlength=max_freq + 1).astype(np.float64)
    total = hist.sum()
    sum_total = (np.arange(len(hist)) * hist).sum()

    sum_b, w_b = 0.0, 0.0
    best_threshold, max_variance = 1, -1.0

    for t in range(1, len(hist)):
        w_b += hist[t - 1]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += (t - 1) * hist[t - 1]
        mean_b = sum_b / w_b
        mean_f = (sum_total - sum_b) / w_f

        between_variance = w_b * w_f * (mean_b - mean_f) ** 2
        if between_variance > max_variance:
            max_variance = between_variance
            best_threshold = t

    return best_threshold


def plot_frequency_cutoff(
    freq_df: pd.DataFrame,
    cutoff: int,
    title: str = "Gene Frequency Cutoff",
) -> go.Figure:
    """
    Histogram of gene frequencies with the cutoff from
    ``compute_frequency_cutoff`` highlighted, so the automatic cut can be
    visually audited before/alongside calling ``summarize_genes``.

    Parameters
    ----------
    freq_df : pd.DataFrame
        Output of ``compute_gene_frequencies``.
    cutoff : int
        Cutoff value, typically the output of ``compute_frequency_cutoff``.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    if freq_df.empty:
        return go.Figure()

    counts = freq_df["Frequency"].value_counts().sort_index()
    colors = ["#b0b0b0" if f < cutoff else "#1f77b4" for f in counts.index]

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index,
                y=counts.values,
                marker_color=colors,
                hovertemplate="Frequency: %{x}<br>Genes: %{y}<extra></extra>",
            )
        ]
    )

    fig.add_vline(
        x=cutoff - 0.5,
        line_dash="dash",
        line_color="red",
        annotation_text=f"cutoff = {cutoff}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Gene frequency (# of cluster pairs)",
        yaxis_title="Number of genes",
        template="plotly_white",
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Biological submatrix + co-occurrence
# ─────────────────────────────────────────────────────────────────────────────

def _build_frequency_plot(df: pd.DataFrame, top_n: Optional[int], label: str) -> go.Figure:
    """Build Plotly frequency bar plot for the selected genes themselves."""
    plot_df = df.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_df["Gene"],
                y=plot_df["Frequency"],
                hovertemplate="Gene: %{x}<br>Frequency: %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"Gene Frequency in {label}",
        xaxis_title="Gene",
        yaxis_title="Frequency",
        template="plotly_white",
    )
    return fig


def _extract_biological_submatrix(
    selected_genes: List[str],
    gene_ids: List[str],
    matrix: np.ndarray,
) -> pd.DataFrame:
    """Extract biological similarity submatrix."""
    gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
    present = [g for g in selected_genes if g in gene_to_idx]
    missing = set(selected_genes) - set(present)

    if missing:
        warnings.warn(f"{len(missing)} genes not found in similarity matrix.")

    if not present:
        return pd.DataFrame()

    idx = [gene_to_idx[g] for g in present]
    sub = matrix[np.ix_(idx, idx)]
    return pd.DataFrame(sub, index=present, columns=present)


def _compute_cooccurrence_jaccard(
    gene_sets: List[List[str]],
    selected_genes: List[str],
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
    selected_genes: List[str]
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
        determined automatically via ``compute_frequency_cutoff`` (Otsu's
        method) — this is the recommended usage, replacing a manually
        guessed threshold.
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
    bio_df = _extract_biological_submatrix(selected_genes, gene_ids, gene_similarity_matrix)

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