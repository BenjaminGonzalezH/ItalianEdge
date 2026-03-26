"""
go_plots.py

This module provides visualization utilities for Gene Ontology (GO)
enrichment results using interactive Plotly charts.

Functions:
1. plot_go_metric:
   Generates a GO plot based on a selected metric.

2. plot_gene_ratio:
   Generates a scatter plot based on gene ratio.

3. plot_qscore:
   Generates a bar plot based on q-score.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Union, Tuple
import logging
import numpy as np
import pandas as pd
import plotly.express as px


logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
MetricType = Literal["gene_ratio", "qscore"]


@dataclass(frozen=True)
class GOPlotOptions:
    """
    Configuration options for GO plots.

    Attributes
    ----------
    colorscale : str
        Plotly colorscale.
    top_n : Optional[int]
        Limit number of displayed terms.
    verbose : bool
        Enable logging.
    use_neglog10 : bool
        Apply -log10 transformation to p-values.
    """
    colorscale: str = "Viridis"
    top_n: Optional[int] = 20
    verbose: bool = True
    use_neglog10: bool = True


def _as_path(p: PathLike) -> Path:
    """
    Convert input into a pathlib.Path object.
    """
    return p if isinstance(p, Path) else Path(p)


def _log(msg: str, verbose: bool):
    """
    Log message if verbose is enabled.
    """
    if verbose:
        logger.info(msg)


def _validate_go_dataframe(df: pd.DataFrame, required_cols: list[str]):
    """
    Validate structure and content of GO dataframe.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame.")

    if df.empty:
        raise ValueError("DataFrame is empty.")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = ["p_value", "gene_ratio", "qscore", "intersection_size"]

    for col in numeric_cols:
        if col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                raise TypeError(f"Column '{col}' must be numeric.")

            if df[col].isna().all():
                raise ValueError(f"Column '{col}' contains only NaN values.")

    if "p_value" in df.columns:
        if (df["p_value"] <= 0).any():
            raise ValueError("p_value must be > 0.")


def _prepare_dataframe(
    df: pd.DataFrame,
    metric: MetricType,
    options: GOPlotOptions
) -> pd.DataFrame:
    """
    Prepare dataframe for plotting.
    """
    df = df.copy()

    # Transform p-values if required
    if options.use_neglog10:
        pvals = df["p_value"].clip(lower=1e-300)
        df["neg_log10_p"] = -np.log10(pvals)
    else:
        df["neg_log10_p"] = df["p_value"]

    # Apply top_n filtering
    if options.top_n is not None:
        sort_col = "p_value" if metric == "gene_ratio" else metric
        ascending = True if metric == "gene_ratio" else False
        df = df.sort_values(sort_col, ascending=ascending).head(options.top_n)

    return df


def _write_html(filepath: PathLike, html: str) -> Path:
    """
    Write HTML content to disk.
    """
    p = _as_path(filepath)

    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

    p.write_text(html, encoding="utf-8")
    return p


def plot_go_metric(
    df: pd.DataFrame,
    metric: MetricType,
    save_path: Optional[PathLike] = None,
    options: GOPlotOptions = GOPlotOptions(),
    return_fig: bool = False,
    return_html: bool = False,
):
    """
    Generate an interactive GO plot using a selected metric.

    Parameters
    ----------
    df : pd.DataFrame
        Enrichment results dataframe.
    metric : {"gene_ratio", "qscore"}
        Metric used for visualization.
    save_path : PathLike, optional
        Path to save HTML output.
    options : GOPlotOptions
        Plot configuration.
    return_fig : bool
        Return Plotly figure.
    return_html : bool
        Return HTML string.

    Returns
    -------
    Optional[Union[Figure, str, Tuple]]
    """
    if metric not in ("gene_ratio", "qscore"):
        raise ValueError("metric must be 'gene_ratio' or 'qscore'")

    required_cols = ["name", metric, "p_value"]
    if metric == "gene_ratio":
        required_cols.append("intersection_size")

    _validate_go_dataframe(df, required_cols)

    df = _prepare_dataframe(df, metric, options)

    if metric == "gene_ratio":
        fig = px.scatter(
            df.sort_values("p_value"),
            x="gene_ratio",
            y="name",
            size="intersection_size",
            color="p_value",
            color_continuous_scale=options.colorscale,
        )
    else:
        fig = px.bar(
            df.sort_values(metric, ascending=False),
            x="qscore",
            y="name",
            color="neg_log10_p",
            color_continuous_scale=options.colorscale,
        )

    html = fig.to_html(include_plotlyjs="cdn", full_html=True)

    if save_path is not None:
        out = _write_html(save_path, html)
        _log(f"[GO plot] HTML saved at: {out}", options.verbose)

    if return_fig and return_html:
        return fig, html
    if return_fig:
        return fig
    if return_html:
        return html
    return None


def plot_gene_ratio(*args, **kwargs):
    return plot_go_metric(*args, metric="gene_ratio", **kwargs)


def plot_qscore(*args, **kwargs):
    return plot_go_metric(*args, metric="qscore", **kwargs)