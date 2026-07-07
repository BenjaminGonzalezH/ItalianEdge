"""
gene_similarity.py

This module provides utilities to compute biological similarity between genes
and clustering solutions using Gene Ontology (GO3-based similarity).

Functions:
1. compute_gene_similarity_matrix_go3:
   Computes a pairwise similarity matrix between genes using GO3.

2. solution_go_similarity_from_dataframe:
   Computes similarity between clustering solutions based on gene similarity.

3. labels_to_clusters:
   Converts label arrays into cluster representations.
"""


from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Set, Tuple, Union, Literal
from itertools import combinations

import logging

import numpy as np
import pandas as pd
import go3


logger = logging.getLogger(__name__)
PathLike = Union[str, Path]

Ontology = Literal["BP", "MF", "CC"]
Groupwise = Literal["bma", "max", "avg", "hausdorff", "simgic"]
SimilarityMeasure = Literal["resnik", "lin", "jc", "simrel", "iccoef", "graphic", "wang", "topoicsim"]
DistanceTransform = Literal["auto", "one_minus", "max_minus", "reciprocal"]
ClusterWeighting = Literal["uniform", "size"]

# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneSimilarityOptions:
    ontology: Ontology = "BP"
    measure: SimilarityMeasure = "wang"
    groupwise: Groupwise = "bma"
    distance_method: DistanceTransform = "auto"
    load_go_terms: bool = True
    num_threads_go3: int = 0  # go3.set_num_threads(0) = auto/por defecto

# -----------------------------------------------------------------------------
# Similitud gen-gen con GO3
# -----------------------------------------------------------------------------

def compute_gene_similarity_matrix_by_batch(
    genes: Sequence[str],
    *,
    obo_path: Union[str, Path],
    gaf_path: Union[str, Path],
    go3_opts: GeneSimilarityOptions = GeneSimilarityOptions(),
)-> Tuple[List[str], np.ndarray]:
    if not isinstance(genes, (list, tuple)) or len(genes) == 0:
        raise ValueError("genes must be a non-empty list/tuple of strings.")

    if go3_opts.ontology not in ("BP", "MF", "CC"):
        raise ValueError("ontology must be 'BP', 'MF' or 'CC'.")
    if go3_opts.groupwise not in ("bma", "max", "avg", "hausdorff", "simgic"):
        raise ValueError("groupwise must be bma, max,avg, hausdorff, simgic.")
    

    go3.set_num_threads(int(go3_opts.num_threads_go3))

    # Cargar GO terms si se pide
    if go3_opts.load_go_terms:
        go3.load_go_terms(str(obo_path))

    # Cargar anotaciones y computar matriz de distancias con GO3
    annotations = go3.load_gaf(str(gaf_path))  # go3 espera path/alias de su loader
    counter = go3.build_term_counter(annotations)

    # Aplicar descarga 
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

    return sim

# -----------------------------------------------------------------------------
# Similitud solución-solución usando SOLO info biológica
# -----------------------------------------------------------------------------

def solution_go_similarity_from_dataframe(
    ids: Sequence[str],
    gene_similarity_matrix: np.ndarray,
    similarity_metric: str,
    reference_df: pd.DataFrame,
    solutions: Sequence[Sequence[Set[str]]],
    *,
    na_value: str = "NA",
    normalize_matrix: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute similarity between clustering solutions using gene-level similarity.

    ... (mismo docstring) ...
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

    for idx, row in reference_df.iterrows():

        s1 = int(row["Solution 1"])
        s2 = int(row["Solution 2"])
        c1 = int(row["Cluster 1"])
        c2 = int(row["Cluster 2"])

        cluster_a = solutions[s1][c1]
        cluster_b = solutions[s2][c2]

        idx_a = [
            id_to_idx[g] for g in cluster_a
            if g != na_value and g in id_to_idx
        ]
        idx_b = [
            id_to_idx[g] for g in cluster_b
            if g != na_value and g in id_to_idx
        ]

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
