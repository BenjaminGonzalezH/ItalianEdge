"""
GO enrichment plotting utilities (refactored professional version).

Features:
- Robust DataFrame validation.
- Deterministic sorting.
- Optional top-N filtering.
- Automatic -log10(p_value) transformation.
- Logging instead of print.
- Optional return_fig / return_html.
- Always saves HTML inside the function (default behavior preserved).
"""

# ──────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Union
import logging
import numpy as np
import pandas as pd
import plotly.express as px


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
MetricType = Literal["gene_ratio", "qscore"]


# ──────────────────────────────────────────────────────────────
# Dataclass Options
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GOPlotOptions:
    """
    Configuration options for GO plots.
    """
    colorscale: str = "Viridis"
    top_n: Optional[int] = 20
    verbose: bool = True
    use_neglog10: bool = True


# ──────────────────────────────────────────────────────────────
# Internal utilities
# ──────────────────────────────────────────────────────────────

def _as_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _log_or_print(msg: str, verbose: bool):
    logger.info(msg)
    if verbose:
        print(msg)


def _validate_go_dataframe(
    df: pd.DataFrame,
    required_cols: list[str]
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame.")

    if df.empty:
        raise ValueError("DataFrame is empty.")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required_cols:
        if col in ["p_value", "gene_ratio", "qscore", "intersection_size"]:
            if not np.issubdtype(df[col].dtype, np.number):
                raise TypeError(f"Column '{col}' must be numeric.")


def _prepare_dataframe(
    df: pd.DataFrame,
    metric: MetricType,
    options: GOPlotOptions
) -> pd.DataFrame:
    df = df.copy()

    if options.use_neglog10 and "p_value" in df.columns:
        df["neg_log10_p"] = -np.log10(df["p_value"].clip(lower=1e-300))
    else:
        df["neg_log10_p"] = df["p_value"]

    if options.top_n is not None:
        df = df.sort_values("p_value", ascending=True).head(options.top_n)

    return df


def _write_html(filepath: PathLike, html: str) -> Path:
    p = _as_path(filepath)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p


# ──────────────────────────────────────────────────────────────
# Main unified plot function
# ──────────────────────────────────────────────────────────────

def plot_go_metric(
    df: pd.DataFrame,
    metric: MetricType,
    save_path: PathLike,
    options: GOPlotOptions = GOPlotOptions(),
    return_fig: bool = False,
    return_html: bool = False,
):
    """
    Generic GO metric plot.

    Always saves HTML to save_path (default project behavior).
    """

    required_cols = ["name", metric, "p_value"]
    if metric == "gene_ratio":
        required_cols.append("intersection_size")

    _validate_go_dataframe(df, required_cols)

    df = _prepare_dataframe(df, metric, options)

    if metric == "gene_ratio":
        fig = px.scatter(
            df.sort_values("p_value", ascending=True),
            x="gene_ratio",
            y="name",
            size="intersection_size",
            color="neg_log10_p",
            color_continuous_scale=options.colorscale,
            title="Gene Ratio for GO Terms",
            labels={
                "gene_ratio": "Gene Ratio",
                "neg_log10_p": "-log10(Adjusted p-value)"
            },
            hover_data=["intersection_size", "p_value"]
        )

    elif metric == "qscore":
        fig = px.bar(
            df.sort_values(metric, ascending=True),
            x="qscore",
            y="name",
            color="neg_log10_p",
            color_continuous_scale=options.colorscale,
            title="Qscore for GO Terms",
            labels={
                "qscore": "Qscore",
                "neg_log10_p": "-log10(Adjusted p-value)"
            }
        )

    else:
        raise ValueError("Unsupported metric.")

    html = fig.to_html(include_plotlyjs="cdn", full_html=True)

    out = _write_html(save_path, html)
    _log_or_print(f"[GO plot] HTML saved at: {out}", options.verbose)

    if return_fig and return_html:
        return fig, html
    if return_fig:
        return fig
    if return_html:
        return html
    return None


# ──────────────────────────────────────────────────────────────
# Public API (backwards semantic compatibility)
# ──────────────────────────────────────────────────────────────

def plot_gene_ratio(
    df: pd.DataFrame,
    save_path: PathLike = "gene_ratioPlot.html",
    options: GOPlotOptions = GOPlotOptions(),
    return_fig: bool = False,
    return_html: bool = False,
):
    return plot_go_metric(
        df,
        metric="gene_ratio",
        save_path=save_path,
        options=options,
        return_fig=return_fig,
        return_html=return_html,
    )


def plot_qscore(
    df: pd.DataFrame,
    save_path: PathLike = "Qplot.html",
    options: GOPlotOptions = GOPlotOptions(),
    return_fig: bool = False,
    return_html: bool = False,
):
    return plot_go_metric(
        df,
        metric="qscore",
        save_path=save_path,
        options=options,
        return_fig=return_fig,
        return_html=return_html,
    )

def plot_go_lollipop(
    df: pd.DataFrame,
    metric: MetricType = "gene_ratio",
    save_path: PathLike = "go_lollipop.html",
    options: GOPlotOptions = GOPlotOptions(),
    return_fig: bool = False,
    return_html: bool = False,
):
    """
    Lollipop plot for GO enrichment.

    X-axis:
        gene_ratio or qscore

    Y-axis:
        GO term name

    Color:
        -log10(p_value)

    Always saves HTML.
    """

    required_cols = ["name", metric, "p_value"]
    if metric == "gene_ratio":
        required_cols.append("intersection_size")

    _validate_go_dataframe(df, required_cols)
    df = _prepare_dataframe(df, metric, options)

    df = df.sort_values("p_value", ascending=True)

    fig = px.scatter(
        df,
        x=metric,
        y="name",
        color="neg_log10_p",
        color_continuous_scale=options.colorscale,
        size=None,
        title=f"Lollipop Plot ({metric.replace('_',' ').title()})",
        labels={
            metric: metric.replace("_", " ").title(),
            "neg_log10_p": "-log10(p-value)"
        },
        hover_data=["p_value"]
    )

    # Add horizontal segments (lollipop sticks)
    for i, row in df.iterrows():
        fig.add_shape(
            type="line",
            x0=0,
            x1=row[metric],
            y0=row["name"],
            y1=row["name"],
            line=dict(width=2, color="rgba(150,150,150,0.5)")
        )

    # Reverse Y axis (most significant on top)
    fig.update_yaxes(autorange="reversed")

    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    out = _write_html(save_path, html)
    _log_or_print(f"[GO lollipop] HTML saved at: {out}", options.verbose)

    if return_fig and return_html:
        return fig, html
    if return_fig:
        return fig
    if return_html:
        return html
    return None