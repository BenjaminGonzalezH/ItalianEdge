import pandas as pd
import plotly.express as px
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

PathLike = Union[str, Path]


@dataclass
class CirGOOptions:

    min_genes_per_term: int = 3
    max_terms: int = 30

    width: int = 900
    height: int = 900


def _gene2terms_to_df(gene2terms):

    rows = []

    for gene, terms in gene2terms.items():

        for term in terms:

            rows.append(
                {
                    "gene": gene,
                    "go_term": term
                }
            )

    return pd.DataFrame(rows)


def _prepare_cirgo_dataframe(df, options):

    gene_counts = (
        df.groupby("go_term")["gene"]
        .nunique()
        .reset_index()
        .rename(columns={"gene": "gene_count"})
    )

    gene_counts = gene_counts[
        gene_counts["gene_count"] >= options.min_genes_per_term
    ]

    gene_counts = gene_counts.sort_values(
        "gene_count",
        ascending=False
    )

    gene_counts = gene_counts.head(options.max_terms)

    gene_counts["category"] = "GO"

    return gene_counts


def plot_cirgo(
    gene2terms: Dict[str, List[str]],
    *,
    options: CirGOOptions = CirGOOptions(),
    save_html_to: Optional[PathLike] = None
):

    df = _gene2terms_to_df(gene2terms)

    cirgo_df = _prepare_cirgo_dataframe(df, options)

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
        title="GO CirGO Visualization"
    )

    if save_html_to:

        p = Path(save_html_to)

        if not p.parent.exists():
            p.parent.mkdir(parents=True)

        fig.write_html(str(p))

    return fig