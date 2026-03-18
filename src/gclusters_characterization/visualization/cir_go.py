"""
cir_go.py

This module provides utilities to visualize Gene Ontology (GO) results
using a CirGO-style sunburst plot.

Functions:
1. plot_cirgo:
   Generates an interactive CirGO visualization from gene-to-term mappings.

2. _gene2terms_to_df:
   Converts gene-to-term dictionary into a tabular format.

3. _prepare_cirgo_dataframe:
   Filters and aggregates GO terms for visualization.
"""

import pandas as pd
import plotly.express as px
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

PathLike = Union[str, Path]


@dataclass
class CirGOOptions:
    """
    Configuration options for CirGO visualization.

    Attributes
    ----------
    min_genes_per_term : int
        Minimum number of genes required for a GO term to be included.
    max_terms : int
        Maximum number of GO terms to display.
    width : int
        Width of the output plot.
    height : int
        Height of the output plot.
    """

    min_genes_per_term: int = 3
    max_terms: int = 30
    width: int = 900
    height: int = 900


def _gene2terms_to_df(gene2terms: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Convert a gene-to-GO-terms mapping into a DataFrame.

    Parameters
    ----------
    gene2terms : Dict[str, List[str]]
        Mapping from gene identifiers to GO terms.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['gene', 'go_term'].
    """
    # Expand mapping into row-wise structure (gene, term)
    rows = []

    for gene, terms in gene2terms.items():
        for term in terms:
            rows.append({"gene": gene, "go_term": term})

    return pd.DataFrame(rows)


def _prepare_cirgo_dataframe(df: pd.DataFrame, options: CirGOOptions) -> pd.DataFrame:
    """
    Prepare aggregated GO term data for CirGO visualization.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing gene-to-term relationships.
    options : CirGOOptions
        Configuration parameters for filtering and display.

    Returns
    -------
    pd.DataFrame
        Aggregated and filtered DataFrame with GO term counts.
    """
    # Count unique genes per GO term
    gene_counts = (
        df.groupby("go_term")["gene"]
        .nunique()
        .reset_index()
        .rename(columns={"gene": "gene_count"})
    )

    # Filter terms based on minimum gene threshold
    gene_counts = gene_counts[
        gene_counts["gene_count"] >= options.min_genes_per_term
    ]

    # Sort terms by importance (gene count)
    gene_counts = gene_counts.sort_values("gene_count", ascending=False)

    # Limit number of displayed terms
    gene_counts = gene_counts.head(options.max_terms)

    # Add artificial root category for hierarchy
    gene_counts["category"] = "GO"

    return gene_counts


def plot_cirgo(
    gene2terms: Dict[str, List[str]],
    *,
    options: CirGOOptions = CirGOOptions(),
    save_html_to: Optional[PathLike] = None
):
    """
    Generate a CirGO-style visualization using a sunburst chart.

    Parameters
    ----------
    gene2terms : Dict[str, List[str]]
        Mapping from genes to GO terms.
    options : CirGOOptions
        Visualization configuration.
    save_html_to : PathLike, optional
        Path to save the interactive HTML plot.

    Returns
    -------
    plotly.graph_objects.Figure
        Generated sunburst figure.
    """

    # Validate input
    if not isinstance(gene2terms, dict):
        raise TypeError("gene2terms must be a dictionary.")

    if not gene2terms:
        raise ValueError("gene2terms cannot be empty.")

    # Convert input mapping to tabular format
    df = _gene2terms_to_df(gene2terms)

    # Aggregate and filter GO terms
    cirgo_df = _prepare_cirgo_dataframe(df, options)

    if cirgo_df.empty:
        raise ValueError("No GO terms passed filtering criteria.")

    # Create sunburst visualization
    fig = px.sunburst(
        cirgo_df,
        path=["category", "go_term"],
        values="gene_count",
        color="gene_count",
        color_continuous_scale="viridis",
        width=options.width,
        height=options.height,
    )

    fig.update_layout(
        title="GO CirGO Visualization",
        margin=dict(t=50, l=0, r=0, b=0)
    )

    # Save interactive HTML if requested
    if save_html_to:
        p = Path(save_html_to)

        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

        fig.write_html(str(p))

    return fig