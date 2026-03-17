"""
gene_similarity.py

Gene Ontology (GO) biological similarity utilities powered by GO3.

Description
-----------
This module computes biological similarity between clustering solutions
using any GO semantic similarity method supported by the GO3 package.

The workflow is:

1. Use a reference DataFrame describing matched cluster pairs.
2. For each matched cluster pair, compute biological similarity from
   all cross-cluster gene pairs.
3. Aggregate cluster-pair similarities into a solution-level matrix.
4. Optionally normalize the final matrix.

Supported GO3 term similarity methods
-------------------------------------
- "resnik"
- "lin"
- "jc"
- "simrel"
- "iccoef"
- "graphic"
- "wang"
- "topoicsim"

Supported groupwise strategies
------------------------------
- "bma"
- "max"
- "avg"
- "hausdorff"
- "simgic"

Functions
---------
1. solution_go_similarity_from_dataframe
   Compute biological similarity from a reference DataFrame using GO3.

2. solution_wang_similarity_from_dataframe
   Backward-compatible wrapper using GO3 Wang similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union
import logging

import numpy as np
import pandas as pd
import go3

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

VALID_GO3_SIMILARITIES = {
    "resnik",
    "lin",
    "jc",
    "simrel",
    "iccoef",
    "graphic",
    "wang",
    "topoicsim",
}

VALID_GROUPWISE = {
    "bma",
    "max",
    "avg",
    "hausdorff",
    "simgic",
}

VALID_ONTOLOGIES = {"BP", "MF", "CC"}

REQUIRED_REFERENCE_COLUMNS = {
    "Solution 1",
    "Solution 2",
    "Cluster 1",
    "Cluster 2",
}


# ------------------------------------------------------------------
# Options
# ------------------------------------------------------------------

@dataclass(frozen=True)
class Go3SimilarityOptions:
    """
    Configuration for GO3-based biological similarity.

    Parameters
    ----------
    similarity : str
        GO3 similarity method.

    groupwise : str
        GO3 groupwise aggregation strategy.

    ontology : str
        GO namespace used for gene comparison.
        Allowed values: "BP", "MF", "CC".

    normalize_matrix : bool
        If True, rescale the final solution-level matrix to [0, 1]
        using the global maximum value.

    na_value : str
        Placeholder used to identify missing genes inside clusters.

    missing_similarity : float
        Value assigned when GO3 cannot compute similarity for a gene pair.

    verbose : bool
        If True, log informative progress messages.
    """

    similarity: str = "wang"
    groupwise: str = "bma"
    ontology: str = "BP"
    normalize_matrix: bool = True
    na_value: str = "NA"
    missing_similarity: float = 0.0
    verbose: bool = False


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _log(msg: str, verbose: bool) -> None:
    """
    Log an internal message and optionally emit it through the logger.

    Parameters
    ----------
    msg : str
        Message to record.

    verbose : bool
        If True, emit the message through the module logger.
    """
    if verbose:
        logger.info(msg)


def _similarity_column_name(similarity: str) -> str:
    """
    Build a human-readable output column name.

    Parameters
    ----------
    similarity : str
        GO3 similarity method name.

    Returns
    -------
    str
        Output column label.
    """
    mapping = {
        "wang": "Wang Similarity",
        "resnik": "Resnik Similarity",
        "lin": "Lin Similarity",
        "jc": "Jiang-Conrath Similarity",
        "simrel": "SimRel Similarity",
        "iccoef": "ICCoef Similarity",
        "graphic": "GraphIC Similarity",
        "topoicsim": "TopoICSim Similarity",
    }
    return mapping.get(similarity, f"{similarity} Similarity")


def _as_str_list(values: Sequence[Union[str, int]]) -> List[str]:
    """
    Convert a sequence of identifiers to a clean list of strings.

    Parameters
    ----------
    values : sequence
        Input identifiers.

    Returns
    -------
    list[str]
        Cleaned identifiers.

    Raises
    ------
    ValueError
        If the input is None.

    TypeError
        If the input is not sequence-like.
    """
    if values is None:
        raise ValueError("ids must not be None.")

    if not isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        raise TypeError(f"ids must be list-like, got: {type(values)}")

    return [str(v).strip() for v in values if v is not None and str(v).strip() != ""]


def _validate_reference_df(reference_df: pd.DataFrame) -> None:
    """
    Validate required columns in the reference DataFrame.

    Parameters
    ----------
    reference_df : pandas.DataFrame
        DataFrame describing matched cluster pairs.

    Raises
    ------
    TypeError
        If reference_df is not a DataFrame.

    ValueError
        If required columns are missing.
    """
    if not isinstance(reference_df, pd.DataFrame):
        raise TypeError("reference_df must be a pandas.DataFrame.")

    missing = REQUIRED_REFERENCE_COLUMNS - set(reference_df.columns)
    if missing:
        raise ValueError(
            f"reference_df is missing required columns: {sorted(missing)}"
        )


def _validate_solutions(
    solutions: Sequence[Sequence[Set[str]]]
) -> None:
    """
    Validate cluster solution structure.

    Parameters
    ----------
    solutions : sequence
        Collection of clustering solutions represented as list of sets.

    Raises
    ------
    TypeError
        If the structure is invalid.
    """
    if not isinstance(solutions, (list, tuple)):
        raise TypeError("solutions must be a list-like collection.")

    for sol in solutions:
        if not isinstance(sol, (list, tuple)):
            raise TypeError("Each solution must be list-like.")
        for cluster in sol:
            if not isinstance(cluster, set):
                raise TypeError("Each cluster must be a set.")


def _validate_options(options: Go3SimilarityOptions) -> None:
    """
    Validate similarity options.

    Parameters
    ----------
    options : Go3SimilarityOptions
        Configuration options.

    Raises
    ------
    ValueError
        If any option is invalid.
    """
    if options.similarity not in VALID_GO3_SIMILARITIES:
        raise ValueError(
            f"Unsupported similarity '{options.similarity}'. "
            f"Choose from {sorted(VALID_GO3_SIMILARITIES)}."
        )

    if options.groupwise not in VALID_GROUPWISE:
        raise ValueError(
            f"Unsupported groupwise '{options.groupwise}'. "
            f"Choose from {sorted(VALID_GROUPWISE)}."
        )

    if options.ontology not in VALID_ONTOLOGIES:
        raise ValueError(
            f"Unsupported ontology '{options.ontology}'. "
            f"Choose from {sorted(VALID_ONTOLOGIES)}."
        )


def _validate_counter(counter) -> None:
    """
    Validate that a GO3 counter object was provided.

    Parameters
    ----------
    counter : object
        GO3 term counter created with go3.build_term_counter().

    Raises
    ------
    ValueError
        If the counter is missing.
    """
    if counter is None:
        raise ValueError(
            "counter must be provided. Build it with "
            "go3.build_term_counter(go3.load_gaf(...))."
        )


def _clean_cluster(
    cluster: Set[str],
    valid_gene_set: Set[str],
    na_value: str,
) -> List[str]:
    """
    Filter a cluster to valid genes only.

    Parameters
    ----------
    cluster : set[str]
        Original cluster.

    valid_gene_set : set[str]
        Allowed gene identifiers.

    na_value : str
        Placeholder representing missing values.

    Returns
    -------
    list[str]
        Cleaned cluster content.
    """
    return sorted(
        str(g)
        for g in cluster
        if g is not None and str(g) != na_value and str(g) in valid_gene_set
    )


def _pairwise_gene_similarity(
    cluster_a: Iterable[str],
    cluster_b: Iterable[str],
    *,
    options: Go3SimilarityOptions,
    counter,
) -> float:
    """
    Compute mean biological similarity between two clusters.

    The similarity is obtained by comparing all cross-cluster gene pairs
    using GO3 compare_genes().

    Parameters
    ----------
    cluster_a : iterable[str]
        First cluster genes.

    cluster_b : iterable[str]
        Second cluster genes.

    options : Go3SimilarityOptions
        Similarity configuration.

    counter : object
        GO3 term counter.

    Returns
    -------
    float
        Mean similarity value across all valid gene pairs.
    """
    values: List[float] = []

    for gene_a in cluster_a:
        for gene_b in cluster_b:
            try:
                sim = go3.compare_genes(
                    gene_a,
                    gene_b,
                    options.ontology,
                    options.similarity,
                    options.groupwise,
                    counter,
                )
                values.append(float(sim))
            except Exception:
                values.append(float(options.missing_similarity))

    if not values:
        return 0.0

    return float(np.mean(values))


def _build_solution_matrix_from_reference(
    reference_df: pd.DataFrame,
    similarity_column: str,
    n_solutions: int,
    normalize_matrix: bool,
) -> np.ndarray:
    """
    Aggregate cluster-level similarities into a solution-level matrix.

    Parameters
    ----------
    reference_df : pandas.DataFrame
        DataFrame with cluster-level similarities.

    similarity_column : str
        Column containing computed biological similarity.

    n_solutions : int
        Number of clustering solutions.

    normalize_matrix : bool
        Whether to normalize the resulting matrix.

    Returns
    -------
    numpy.ndarray
        Symmetric solution-level similarity matrix.
    """
    matrix = np.zeros((n_solutions, n_solutions), dtype=float)
    np.fill_diagonal(matrix, 1.0)

    grouped = (
        reference_df.groupby(["Solution 1", "Solution 2"])[similarity_column]
        .mean()
        .reset_index()
    )

    for _, row in grouped.iterrows():
        s1 = int(row["Solution 1"])
        s2 = int(row["Solution 2"])
        sim = float(row[similarity_column])

        matrix[s1, s2] = sim
        matrix[s2, s1] = sim

    if normalize_matrix:
        max_value = float(np.nanmax(matrix))
        if max_value > 0:
            matrix = matrix / max_value

    np.fill_diagonal(matrix, 1.0)
    return matrix


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def solution_go_similarity_from_dataframe(
    ids: Sequence[Union[str, int]],
    reference_df: pd.DataFrame,
    solutions: Sequence[Sequence[Set[str]]],
    *,
    counter,
    options: Go3SimilarityOptions = Go3SimilarityOptions(),
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute GO-based biological similarity from a reference DataFrame.

    This function reads a DataFrame that describes matched cluster pairs
    between clustering solutions. For each matched cluster pair, it computes
    biological similarity using GO3 and appends the result to the DataFrame.

    A solution-level similarity matrix is then built by averaging the
    cluster-level similarities for each solution pair.

    Parameters
    ----------
    ids : sequence[str or int]
        Valid gene identifiers.

    reference_df : pandas.DataFrame
        DataFrame describing cluster matches. Required columns:

        - "Solution 1"
        - "Solution 2"
        - "Cluster 1"
        - "Cluster 2"

    solutions : sequence
        Clustering solutions represented as list-of-set clusters.

    counter : object
        GO3 term counter produced by go3.build_term_counter().

    options : Go3SimilarityOptions
        GO similarity configuration.

    Returns
    -------
    tuple
        matrix : numpy.ndarray
            Symmetric solution-level similarity matrix.

        dataframe : pandas.DataFrame
            Reference DataFrame with an appended similarity column.
    """
    ids_clean = _as_str_list(ids)
    valid_gene_set = set(ids_clean)

    _validate_reference_df(reference_df)
    _validate_solutions(solutions)
    _validate_options(options)
    _validate_counter(counter)

    similarity_column = _similarity_column_name(options.similarity)

    df = reference_df.copy()
    similarities: List[float] = []

    _log(
        f"[GO3] Computing {options.similarity} similarities with "
        f"groupwise={options.groupwise} and ontology={options.ontology}.",
        options.verbose,
    )

    for _, row in df.iterrows():
        s1 = int(row["Solution 1"])
        s2 = int(row["Solution 2"])
        c1 = int(row["Cluster 1"])
        c2 = int(row["Cluster 2"])

        cluster_a = _clean_cluster(solutions[s1][c1], valid_gene_set, options.na_value)
        cluster_b = _clean_cluster(solutions[s2][c2], valid_gene_set, options.na_value)

        if len(cluster_a) == 0 or len(cluster_b) == 0:
            sim = 0.0
        else:
            sim = _pairwise_gene_similarity(
                cluster_a,
                cluster_b,
                options=options,
                counter=counter,
            )

        similarities.append(sim)

    df[similarity_column] = similarities

    matrix = _build_solution_matrix_from_reference(
        df,
        similarity_column=similarity_column,
        n_solutions=len(solutions),
        normalize_matrix=options.normalize_matrix,
    )

    return matrix, df

def compute_go_gene_similarity_matrix(
    genes: Sequence[str],
    *,
    counter,
    similarity: str = "wang",
    groupwise: str = "bma",
    ontology: str = "BP",
    as_distance: bool = False,
    missing_similarity: float = 0.0,
) -> np.ndarray:
    """
    Compute pairwise GO3 similarity (or distance) matrix using batch API.

    This implementation leverages GO3 batch computation for efficiency.

    Parameters
    ----------
    genes : list[str]
        Gene identifiers.

    counter : object
        GO3 term counter.

    similarity : str
        GO similarity method.

    groupwise : str
        Aggregation method.

    ontology : str
        GO namespace ("BP", "MF", "CC").

    as_distance : bool
        If True, return distance matrix (1 - similarity).

    missing_similarity : float
        Fallback value if computation fails.

    Returns
    -------
    np.ndarray
        Symmetric similarity (or distance) matrix.
    """

    if counter is None:
        raise ValueError("counter must not be None.")

    if not isinstance(genes, (list, tuple, np.ndarray)):
        raise TypeError("genes must be list-like.")

    genes = [str(g) for g in genes if g is not None and str(g) != ""]
    n = len(genes)

    if n == 0:
        raise ValueError("genes list is empty.")

    # --------------------------------------------------
    # Build upper triangle pairs
    # --------------------------------------------------

    pairs = []
    indices = []

    for i in range(n):
        for j in range(i, n):
            pairs.append((genes[i], genes[j]))
            indices.append((i, j))

    # --------------------------------------------------
    # Compute batch similarities
    # --------------------------------------------------

    try:
        sims = go3.compare_gene_pairs_batch(
            pairs,
            ontology,
            similarity,
            groupwise,
            counter,
        )
    except Exception:
        sims = [missing_similarity] * len(pairs)

    # --------------------------------------------------
    # Build matrix
    # --------------------------------------------------

    matrix = np.zeros((n, n), dtype=float)

    for (i, j), sim in zip(indices, sims):

        try:
            sim = float(sim)
        except Exception:
            sim = missing_similarity

        matrix[i, j] = sim
        matrix[j, i] = sim

    # Force diagonal = 1
    np.fill_diagonal(matrix, 1.0)

    # Convert to distance if needed
    if as_distance:
        matrix = 1.0 - matrix

    return matrix