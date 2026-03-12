"""
GoChordDiagram.py

Interactive Gene–GO chord diagram visualization.

Features
--------
- Accepts gene -> GO terms structure
- Automatic edge construction
- Filtering of highly connected nodes
- HTML export
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import holoviews as hv
from holoviews import opts
import logging

hv.extension("bokeh")

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


# ─────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class GoChordOptions:

    max_terms_per_gene: Optional[int] = None
    min_gene_frequency: int = 1
    width: int = 900
    height: int = 900
    verbose: bool = True


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _log(msg: str, verbose: bool):

    logger.info(msg)
    if verbose:
        print(msg)


def _validate_input(gene2terms: Dict[str, List[str]]):

    if not isinstance(gene2terms, dict):
        raise TypeError("gene2terms must be dict[str, list[str]]")

    if not gene2terms:
        raise ValueError("gene2terms cannot be empty")


# ─────────────────────────────────────────────
# Conversion utilities
# ─────────────────────────────────────────────

def _gene2terms_to_dataframe(
    gene2terms: Dict[str, List[str]],
    options: GoChordOptions
) -> pd.DataFrame:

    rows = []

    for gene, terms in gene2terms.items():

        if options.max_terms_per_gene:
            terms = terms[: options.max_terms_per_gene]

        for term in terms:

            rows.append({
                "gene": str(gene),
                "go_term": str(term)
            })

    df = pd.DataFrame(rows)

    if options.min_gene_frequency > 1:

        counts = df["go_term"].value_counts()

        keep_terms = counts[counts >= options.min_gene_frequency].index

        df = df[df["go_term"].isin(keep_terms)]

    return df


# ─────────────────────────────────────────────
# Chord construction
# ─────────────────────────────────────────────

def _build_chord(df: pd.DataFrame, options: GoChordOptions):

    genes = df["gene"].unique()
    terms = df["go_term"].unique()

    nodes = list(genes) + list(terms)

    node_df = pd.DataFrame({
        "index": range(len(nodes)),
        "name": nodes
    })

    node_map = dict(zip(nodes, node_df.index))

    edges = pd.DataFrame({
        "source": df["gene"].map(node_map),
        "target": df["go_term"].map(node_map),
        "value": 1
    })

    chord = hv.Chord((edges, hv.Dataset(node_df, "index")))

    chord.opts(
        opts.Chord(
            cmap="Category20",
            labels="name",
            node_color="name",
            edge_color="source",
            width=options.width,
            height=options.height,
        )
    )

    return chord


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def plot_go_chord_html(
    gene2terms: Dict[str, List[str]],
    *,
    options: GoChordOptions = GoChordOptions(),
    save_html_to: Optional[PathLike] = None,
    return_plot: bool = False,
):

    _validate_input(gene2terms)

    df = _gene2terms_to_dataframe(gene2terms, options)

    if df.empty:
        raise ValueError("No edges available after filtering.")

    chord = _build_chord(df, options)

    if save_html_to:

        p = Path(save_html_to)

        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

        hv.save(chord, str(p))

        _log(f"[GO] Chord diagram saved at: {p}", options.verbose)

    if return_plot:
        return chord

    return None