"""
Disjoint Genes Extraction Module.

This module provides utilities to compute the symmetric difference (disjoint genes)
between equivalent cluster pairs obtained from different clustering solutions.

The main goal is to identify the genes that explain the biological or structural
differences between clusters already filtered by a similarity threshold.

Functions
---------
1. compute_disjoint_genes_dataframe:
   Generates a new DataFrame including disjoint gene sets for each valid pair.
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


def summarize_disjoint_genes(
    df: pd.DataFrame,
    gene_column: str,
    gene_ids: Sequence[str],
    gene_similarity_matrix: np.ndarray,
    *,
    min_gene_frequency: int = 1,
    top_n_genes_for_plot: int | None = 30,
) -> Tuple[pd.DataFrame, go.Figure, pd.DataFrame, pd.DataFrame]:
    """
    Summarize genes appearing in disjoint gene sets.

    This function:
    1. Counts gene frequencies across disjoint sets
    2. Generates a frequency bar chart
    3. Extracts a biological submatrix for selected genes
    4. Computes a gene co-occurrence matrix

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing disjoint gene collections.

    gene_column : str
        Name of the column containing iterable gene collections
        (e.g., sets, lists, or tuples).

    gene_ids : Sequence[str]
        Ordered list of gene identifiers corresponding to the rows and columns
        of `gene_similarity_matrix`.

    gene_similarity_matrix : np.ndarray
        Square biological similarity matrix aligned with `gene_ids`.

    min_gene_frequency : int, optional
        Minimum number of occurrences required for a gene to be retained in the
        output matrices.

    top_n_genes_for_plot : int | None, optional
        Maximum number of genes to display in the frequency plot. If None,
        all selected genes are displayed.

    Returns
    -------
    Tuple[pd.DataFrame, go.Figure, pd.DataFrame, pd.DataFrame]
        A tuple containing:

        1. gene_frequency_df
            DataFrame with columns:
            - "Gene"
            - "Frequency"

        2. frequency_figure
            Plotly bar chart showing gene frequencies.

        3. biological_submatrix_df
            DataFrame whose index and columns are gene identifiers and whose
            values correspond to the biological similarity submatrix.

        4. cooccurrence_matrix_df
            DataFrame whose index and columns are gene identifiers and whose
            values correspond to the co-occurrence counts across disjoint sets.
    """
    required_columns = {gene_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    gene_ids = list(map(str, gene_ids))

    if gene_similarity_matrix.ndim != 2:
        raise ValueError("gene_similarity_matrix must be a 2D array.")

    if gene_similarity_matrix.shape[0] != gene_similarity_matrix.shape[1]:
        raise ValueError("gene_similarity_matrix must be square.")

    if gene_similarity_matrix.shape[0] != len(gene_ids):
        raise ValueError(
            "The size of gene_similarity_matrix must match the length of gene_ids."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Normalize and flatten gene sets
    # ──────────────────────────────────────────────────────────────────────────
    normalized_gene_sets: list[list[str]] = []

    for genes in df[gene_column]:
        if isinstance(genes, (set, list, tuple)):
            normalized = [str(g).strip() for g in genes if pd.notna(g)]
            normalized = [g for g in normalized if g != ""]
            normalized_gene_sets.append(normalized)

    all_genes = [gene for gene_list in normalized_gene_sets for gene in gene_list]

    if not all_genes:
        empty_freq_df = pd.DataFrame(columns=["Gene", "Frequency"])
        empty_fig = go.Figure()
        empty_bio_df = pd.DataFrame()
        empty_cooc_df = pd.DataFrame()
        return empty_freq_df, empty_fig, empty_bio_df, empty_cooc_df

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Gene frequencies
    # ──────────────────────────────────────────────────────────────────────────
    gene_counter = Counter(all_genes)

    gene_frequency_df = (
        pd.DataFrame(
            [{"Gene": gene, "Frequency": count} for gene, count in gene_counter.items()]
        )
        .sort_values(by=["Frequency", "Gene"], ascending=[False, True])
        .reset_index(drop=True)
    )

    selected_genes = gene_frequency_df.loc[
        gene_frequency_df["Frequency"] >= min_gene_frequency, "Gene"
    ].tolist()

    if not selected_genes:
        empty_fig = go.Figure()
        empty_bio_df = pd.DataFrame()
        empty_cooc_df = pd.DataFrame()
        return gene_frequency_df, empty_fig, empty_bio_df, empty_cooc_df

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Frequency figure
    # ──────────────────────────────────────────────────────────────────────────
    plot_df = gene_frequency_df[
        gene_frequency_df["Gene"].isin(selected_genes)
    ].copy()

    if top_n_genes_for_plot is not None:
        plot_df = plot_df.head(top_n_genes_for_plot)

    frequency_figure = go.Figure(
        data=[
            go.Bar(
                x=plot_df["Gene"],
                y=plot_df["Frequency"],
                hovertemplate="Gene: %{x}<br>Frequency: %{y}<extra></extra>",
            )
        ]
    )

    frequency_figure.update_layout(
        title="Gene Frequency in Disjoint Gene Sets",
        xaxis_title="Gene",
        yaxis_title="Frequency",
        template="plotly_white",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Biological submatrix
    # ──────────────────────────────────────────────────────────────────────────
    gene_to_index = {gene: idx for idx, gene in enumerate(gene_ids)}
    selected_genes_in_matrix = [g for g in selected_genes if g in gene_to_index]

    if selected_genes_in_matrix:
        indices = [gene_to_index[g] for g in selected_genes_in_matrix]
        biological_submatrix = gene_similarity_matrix[np.ix_(indices, indices)]

        biological_submatrix_df = pd.DataFrame(
            biological_submatrix,
            index=selected_genes_in_matrix,
            columns=selected_genes_in_matrix,
        )
    else:
        biological_submatrix_df = pd.DataFrame()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Co-occurrence matrix (normalized)
    # ──────────────────────────────────────────────────────────────────────────
    selected_gene_set = set(selected_genes)
    ordered_genes_for_cooc = [g for g in selected_genes if g in selected_gene_set]

    # Initialize count matrix
    cooccurrence_matrix = pd.DataFrame(
        0,
        index=ordered_genes_for_cooc,
        columns=ordered_genes_for_cooc,
        dtype=int,
    )

    # Count co-occurrences
    for gene_list in normalized_gene_sets:
        filtered_list = list(set(g for g in gene_list if g in selected_gene_set))

        for i, gene_a in enumerate(filtered_list):
            cooccurrence_matrix.loc[gene_a, gene_a] += 1

            for gene_b in filtered_list[i + 1:]:
                cooccurrence_matrix.loc[gene_a, gene_b] += 1
                cooccurrence_matrix.loc[gene_b, gene_a] += 1

    # ──────────────────────────────────────────────────────────────────────────
    # Normalize (Jaccard similarity)
    # ──────────────────────────────────────────────────────────────────────────
    cooccurrence_matrix = cooccurrence_matrix.astype(float)

    # Frecuencia (diagonal)
    freq_dict = {
        gene: cooccurrence_matrix.loc[gene, gene]
        for gene in cooccurrence_matrix.index
    }

    cooccurrence_matrix_prop = cooccurrence_matrix.copy()

    for g1 in cooccurrence_matrix_prop.index:
        for g2 in cooccurrence_matrix_prop.columns:

            c_ij = cooccurrence_matrix.loc[g1, g2]
            f_i = freq_dict[g1]
            f_j = freq_dict[g2]

            denom = f_i + f_j - c_ij

            if denom > 0:
                cooccurrence_matrix_prop.loc[g1, g2] = c_ij / denom
            else:
                cooccurrence_matrix_prop.loc[g1, g2] = 0.0

    cooccurrence_matrix_df = cooccurrence_matrix_prop

    return (
        gene_frequency_df,
        frequency_figure,
        biological_submatrix_df,
        cooccurrence_matrix_df,
    )