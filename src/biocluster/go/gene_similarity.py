"""
gene_similarity: This module provides utilities to compute biological similarity between genes
and clustering solutions using Gene Ontology (GO3-based similarity).

Functions:
1. compute_gene_similarity_matrix_by_batch - Computes a pairwise similarity matrix between genes using GO3.
2. solution_go_similarity_from_dataframe   - Computes similarity between clustering solutions based on gene similarity.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal, Union

import go3
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]

Ontology = Literal["BP", "MF", "CC"]
Groupwise = Literal["bma", "max", "avg", "hausdorff", "simgic"]
SimilarityMeasure = Literal[
    "resnik", "lin", "jc", "simrel", "iccoef", "graphic", "wang", "topoicsim"
]
DistanceTransform = Literal["auto", "one_minus", "max_minus", "reciprocal"]

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneSimilarityOptions:
    ontology: Ontology = "BP"
    measure: SimilarityMeasure = "wang"
    groupwise: Groupwise = "bma"
    distance_method: DistanceTransform = "auto"
    load_go_terms: bool = True
    num_threads_go3: int = 0  # go3.set_num_threads(0) = auto/default


# -----------------------------------------------------------------------------
# Gene-by-gene similarity
# -----------------------------------------------------------------------------


def compute_gene_similarity_matrix_by_batch(
    genes: Sequence[str],
    *,
    obo_path: Union[str, Path],
    gaf_path: Union[str, Path],
    go3_opts: GeneSimilarityOptions = GeneSimilarityOptions(),
) -> tuple[list[str], np.ndarray]:
    if not isinstance(genes, (list, tuple)) or len(genes) == 0:
        raise ValueError("genes must be a non-empty list/tuple of strings.")

    if go3_opts.ontology not in ("BP", "MF", "CC"):
        raise ValueError("ontology must be 'BP', 'MF' or 'CC'.")
    if go3_opts.groupwise not in ("bma", "max", "avg", "hausdorff", "simgic"):
        raise ValueError("groupwise must be bma, max,avg, hausdorff, simgic.")

    go3.set_num_threads(int(go3_opts.num_threads_go3))

    # Load GO terms if requested
    if go3_opts.load_go_terms:
        go3.load_go_terms(str(obo_path))

    # Load annotations and build the term counter used by GO3's similarity functions
    annotations = go3.load_gaf(str(gaf_path))  # go3 expects a path/alias for its loader
    counter = go3.build_term_counter(annotations)

    # Compute pairwise similarity scores for all gene combinations
    pairs = list(combinations(genes, 2))
    scores = go3.compare_gene_pairs_batch(
        pairs,
        ontology=go3_opts.ontology,
        similarity=go3_opts.measure,
        groupwise=go3_opts.groupwise,
        counter=counter,
    )

    genes_list = list(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes_list)}
    n = len(genes_list)
    sim = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(sim, 1.0)

    for (g1, g2), score in zip(pairs, scores):
        i = gene_to_idx[g1]
        j = gene_to_idx[g2]
        sim[i, j] = float(score)
        sim[j, i] = float(score)

    return genes_list, sim


# -----------------------------------------------------------------------------
# Solution-to-solution similarity using ONLY biological information
# -----------------------------------------------------------------------------


def solution_go_similarity_from_dataframe(
    ids: Sequence[str],
    gene_similarity_matrix: np.ndarray,
    similarity_metric: str,
    reference_df: pd.DataFrame,
    solutions: Sequence[Sequence[set[str]]],
    *,
    na_value: str = "NA",
    normalize_matrix: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute similarity between clustering solutions using gene-level similarity.

    For every matched cluster pair listed in ``reference_df`` (as produced by
    e.g. ``find_equivalent_clusters_jaccard``/``find_equivalent_clusters_rand``),
    this computes the mean pairwise gene similarity between the two clusters
    using ``gene_similarity_matrix``, then aggregates those pair-level scores
    into a symmetric solution-to-solution similarity matrix.

    Parameters
    ----------
    ids : Sequence[str]
        Gene identifiers indexing the rows/columns of ``gene_similarity_matrix``.
    gene_similarity_matrix : numpy.ndarray
        Precomputed pairwise gene similarity matrix (n_genes x n_genes).
    similarity_metric : str
        Name used to label the new similarity column added to ``reference_df``.
    reference_df : pandas.DataFrame
        Table of matched clusters with columns "Solution 1", "Solution 2",
        "Cluster 1", "Cluster 2" (one row per matched cluster pair).
    solutions : Sequence[Sequence[Set[str]]]
        Clustering solutions; each solution is a sequence of gene sets.
    na_value : str, default "NA"
        Placeholder value to exclude when looking up genes in ``ids``.
    normalize_matrix : bool, default True
        If True, scales the resulting matrix so its maximum value is 1.0.

    Returns
    -------
    Tuple[numpy.ndarray, pandas.DataFrame]
        The symmetric solution-to-solution similarity matrix, and a copy of
        ``reference_df`` with an added ``f"{similarity_metric} Similarity"``
        column holding the per-pair similarity score.
    """

    required_cols = {
        "Solution 1",
        "Solution 2",
        "Cluster 1",
        "Cluster 2",
    }
    if not required_cols.issubset(reference_df.columns):
        raise ValueError(f"reference_df must contain columns: {required_cols}")

    n = len(solutions)
    final_matrix = np.zeros((n, n), dtype=np.float64)
    pair_counts = np.zeros((n, n), dtype=np.float64)

    ids_list = [str(x) for x in ids]
    id_to_idx = {g: i for i, g in enumerate(ids_list)}

    similarity_values = []

    for _idx, row in reference_df.iterrows():

        s1 = int(row["Solution 1"])
        s2 = int(row["Solution 2"])
        c1 = int(row["Cluster 1"])
        c2 = int(row["Cluster 2"])

        cluster_a = solutions[s1][c1]
        cluster_b = solutions[s2][c2]

        idx_a = [id_to_idx[g] for g in cluster_a if g != na_value and g in id_to_idx]
        idx_b = [id_to_idx[g] for g in cluster_b if g != na_value and g in id_to_idx]

        if not idx_a or not idx_b:
            similarity = 0.0
        else:
            submatrix = gene_similarity_matrix[np.ix_(idx_a, idx_b)]
            similarity = float(np.nanmean(submatrix))

        similarity_values.append(similarity)

        final_matrix[s1, s2] += similarity
        final_matrix[s2, s1] += similarity
        pair_counts[s1, s2] += 1
        pair_counts[s2, s1] += 1

    nonzero = pair_counts > 0
    final_matrix[nonzero] /= pair_counts[nonzero]

    reference_df = reference_df.copy()
    reference_df[f"{similarity_metric} Similarity"] = similarity_values

    if normalize_matrix and np.max(final_matrix) > 0:
        final_matrix /= np.max(final_matrix)

    np.fill_diagonal(final_matrix, 1.0)

    return final_matrix, reference_df
