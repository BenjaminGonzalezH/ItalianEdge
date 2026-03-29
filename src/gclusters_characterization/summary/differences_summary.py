"""
differences_summary.py

Utility functions to compute the symmetric difference (disjoint genes)
between equivalent cluster pairs obtained from different clustering solutions.

The main goal is to identify the genes that explain the biological or structural
differences between clusters already filtered by a similarity threshold.

Functions
---------
1. compute_disjoint_genes_dataframe:
   Generates a new DataFrame including disjoint gene sets for each valid pair.
2. summarize_disjoint_genes:
   Summarize gene frequencies, biological submatrix, and co-occurrence matrix.
"""
# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
from typing import Dict, Tuple, Set
import pandas as pd
from typing import Dict, Sequence
from collections import Counter
import numpy as np
import plotly.graph_objects as go
import warnings
from typing import Sequence, Tuple, List

# ──────────────────────────────────────────────────────────────────────────────
# Core Function
# ──────────────────────────────────────────────────────────────────────────────
def compute_disjoint_genes_dataframe(
    df: pd.DataFrame,
    metric_col: str,
    solution_cluster_matrix: Dict[Tuple[str, str], Set[str]],
    threshold: float,
) -> pd.DataFrame:
    """
    Compute disjoint genes for cluster pairs under a given threshold.

    This function filters cluster pairs based on a metric threshold and computes
    the symmetric difference (disjoint genes) between the gene sets of each pair.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing cluster pair relationships. Must include:
        - "Solution 1"
        - "Solution 2"
        - "Cluster 1"
        - "Cluster 2"
        - metric_col

    metric_col : str
        Name of the column containing the similarity/difference metric used
        for threshold filtering.

    solution_cluster_matrix : Dict[Tuple[str, str], Set[str]]
        Mapping from (solution_id, cluster_id) → set of genes.

        Example:
            {
                ("S1", "C1"): {"geneA", "geneB"},
                ("S2", "C3"): {"geneB", "geneC"},
            }

    threshold : float
        Threshold used to filter relevant pairs.
        Only pairs where df[metric_col] <= threshold are considered.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing:
        - original pair identifiers
        - disjoint gene sets (as Python sets)

        Note:
        The output does NOT include gene cardinality (size), only the gene sets.
    """

    required_columns = {
        "Solution 1",
        "Solution 2",
        "Cluster 1",
        "Cluster 2",
        metric_col,
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ──────────────────────────────────────────────────────────────────────────
    # Filter pairs by threshold
    # ──────────────────────────────────────────────────────────────────────────
    filtered_df = df[df[metric_col] <= threshold].copy()

    # ──────────────────────────────────────────────────────────────────────────
    # Compute disjoint genes
    # ──────────────────────────────────────────────────────────────────────────
    disjoint_genes_list = []

    for _, row in filtered_df.iterrows():

        genes_1 = solution_cluster_matrix[int(row["Solution 1"])][int(row["Cluster 1"])]
        genes_2 = solution_cluster_matrix[int(row["Solution 2"])][int(row["Cluster 2"])]

        # Symmetric difference (disjoint genes)
        disjoint = (genes_1 - genes_2) | (genes_2 - genes_1)

        disjoint_genes_list.append(disjoint)

    # ──────────────────────────────────────────────────────────────────────────
    # Build output DataFrame
    # ──────────────────────────────────────────────────────────────────────────
    output_df = filtered_df[
        ["Solution 1", "Cluster 1", "Solution 2", "Cluster 2"]
    ].copy()

    output_df["Disjoint Genes"] = disjoint_genes_list

    return output_df



# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def _compute_gene_frequencies(gene_sets: List[List[str]]) -> pd.DataFrame:
    """Compute gene frequencies."""
    all_genes = [g for genes in gene_sets for g in genes]

    if not all_genes:
        return pd.DataFrame(columns=["Gene", "Frequency"])

    counter = Counter(all_genes)

    return (
        pd.DataFrame(
            [{"Gene": g, "Frequency": c} for g, c in counter.items()]
        )
        .sort_values(["Frequency", "Gene"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _build_frequency_plot(df: pd.DataFrame, top_n: int | None) -> go.Figure:
    """Build Plotly frequency bar plot."""
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
        title="Gene Frequency in Disjoint Gene Sets",
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

    # ─────────────────────────────────────
    # Build binary matrix (sets × genes)
    # ─────────────────────────────────────
    X = np.zeros((n_sets, n_genes), dtype=np.uint64)

    for i, genes in enumerate(gene_sets):
        for g in set(genes):
            if g in gene_to_idx:
                X[i, gene_to_idx[g]] = 1

    # ─────────────────────────────────────
    # Co-occurrence (fast BLAS)
    # ─────────────────────────────────────
    C = X.T @ X

    # ─────────────────────────────────────
    # Jaccard normalization
    # ─────────────────────────────────────
    diag = np.diag(C)
    denom = diag[:, None] + diag[None, :] - C

    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(denom > 0, C / denom, 0.0)

    return pd.DataFrame(J, index=selected_genes, columns=selected_genes)


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def summarize_disjoint_genes(
    df: pd.DataFrame,
    gene_column: str,
    gene_ids: Sequence[str],
    gene_similarity_matrix: np.ndarray,
    *,
    min_gene_frequency: int = 1,
    top_n_genes_for_plot: int | None = 30,
    max_genes: int | None = None,
) -> Tuple[pd.DataFrame, go.Figure, pd.DataFrame, pd.DataFrame]:
    """
    Summarize genes appearing in disjoint gene sets.

    Refactored version with vectorized co-occurrence computation.

    Returns
    -------
    gene_frequency_df
    frequency_figure
    biological_submatrix_df
    cooccurrence_jaccard_df
    """

    # ─────────────────────────────────────
    # Validation
    # ─────────────────────────────────────
    if gene_column not in df.columns:
        raise ValueError(f"Missing column: {gene_column}")

    gene_ids = list(map(str, gene_ids))

    if gene_similarity_matrix.ndim != 2:
        raise ValueError("gene_similarity_matrix must be 2D")

    if gene_similarity_matrix.shape[0] != gene_similarity_matrix.shape[1]:
        raise ValueError("gene_similarity_matrix must be square")

    if gene_similarity_matrix.shape[0] != len(gene_ids):
        raise ValueError("Matrix size must match gene_ids length")

    # ─────────────────────────────────────
    # Step 1: normalize
    # ─────────────────────────────────────
    gene_sets = _normalize_gene_sets(df, gene_column)

    if not gene_sets:
        return (
            pd.DataFrame(columns=["Gene", "Frequency"]),
            go.Figure(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    # ─────────────────────────────────────
    # Step 2: frequency
    # ─────────────────────────────────────
    freq_df = _compute_gene_frequencies(gene_sets)

    selected_genes = freq_df.loc[
        freq_df["Frequency"] >= min_gene_frequency, "Gene"
    ].tolist()

    if max_genes is not None:
        selected_genes = selected_genes[:max_genes]

    if not selected_genes:
        return freq_df, go.Figure(), pd.DataFrame(), pd.DataFrame()

    # ─────────────────────────────────────
    # Step 3: plot
    # ─────────────────────────────────────
    fig = _build_frequency_plot(
        freq_df[freq_df["Gene"].isin(selected_genes)],
        top_n_genes_for_plot,
    )

    # ─────────────────────────────────────
    # Step 4: biological matrix
    # ─────────────────────────────────────
    bio_df = _extract_biological_submatrix(
        selected_genes,
        gene_ids,
        gene_similarity_matrix,
    )

    # ─────────────────────────────────────
    # Step 5: co-occurrence (vectorized)
    # ─────────────────────────────────────
    cooc_df = _compute_cooccurrence_jaccard(
        gene_sets,
        selected_genes,
    )

    return freq_df, fig, bio_df, cooc_df