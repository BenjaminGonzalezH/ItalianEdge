"""
GoNetwork: GO term interaction network based on Wang semantic similarity.

Features:
- No pygosemsim dependency
- Wang similarity computed via go3
- Deterministic layout
- Interactive HTML export
- Clean, validated architecture

Functions
1. plot_go_interaction_network_html – Build and export a GO term similarity network as an interactive figure/HTML.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
from __future__  import annotations
from dataclasses import dataclass
from pathlib     import Path
from typing      import Dict, List, Optional, Sequence, Union
from itertools   import combinations
import networkx             as nx
import plotly.graph_objects as go
import matplotlib.colors    as mcolors
import go3
import logging
import math

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


# ──────────────────────────────────────────────────────────────
# Options
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GoNetworkOptions:
    similarity_threshold: float = 0.7
    min_genes_per_term: int = 1
    max_node_size: float = 40.0
    layout: str = "spring"  # "spring" or "kamada_kawai"
    random_state: int = 42
    verbose: bool = False


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def _log(msg: str, verbose: bool):
    logger.info(msg)
    if verbose:
        print(msg)


def _validate_inputs(gene2terms: Dict[str, List[str]], term_pvalues: Dict[str, float]):
    if not isinstance(gene2terms, dict) or not gene2terms:
        raise ValueError("gene2terms must be a non-empty dictionary.")

    if not isinstance(term_pvalues, dict):
        raise ValueError("term_pvalues must be a dictionary.")


def _count_terms(gene2terms: Dict[str, List[str]]) -> Dict[str, int]:
    term_counts = {}
    for gene, terms in gene2terms.items():
        for term in terms:
            term_counts.setdefault(term, 0)
            term_counts[term] += 1
    return term_counts


def _build_similarity_graph(
    term_list: Sequence[str],
    counter,
    threshold: float,
    verbose: bool,
) -> nx.Graph:

    G = nx.Graph()

    for t in term_list:
        G.add_node(t)

    for t1, t2 in combinations(term_list, 2):
        try:
            sim = go3.semantic_similarity(t1, t2, "wang", counter)
            if sim >= threshold:
                G.add_edge(t1, t2, weight=float(sim))
        except Exception:
            continue

    _log(f"[GO] Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.", verbose)
    return G


def _compute_layout(G: nx.Graph, layout: str, random_state: int):

    if layout == "spring":
        return nx.spring_layout(G, weight="weight", seed=random_state)

    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G)

    raise ValueError("layout must be 'spring' or 'kamada_kawai'")


def _build_plotly_figure(
    G: nx.Graph,
    pos: Dict,
    term_counts: Dict[str, int],
    term_pvalues: Dict[str, float],
    max_node_size: float,
):

    if G.number_of_nodes() == 0:
        return None

    # Color mapping by p-value
    cmap = mcolors.LinearSegmentedColormap.from_list("pvalue", ["red", "yellow", "green"])
    pvalues = [term_pvalues.get(t, 0.05) for t in G.nodes()]
    norm = mcolors.Normalize(vmin=min(pvalues), vmax=max(pvalues))

    def get_color(term):
        p = term_pvalues.get(term, None)

        if p is None or (isinstance(p, float) and math.isnan(p)):
            return "#808080"  # color for missing values
  
        return mcolors.to_hex(cmap(norm(term_pvalues.get(term, 0.05))))

    # Edges
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="gray"),
        hoverinfo="none"
    )

    # Nodes
    node_x = []
    node_y = []
    node_sizes = []
    node_colors = []
    hover_text = []

    max_count = max(term_counts.values()) if term_counts else 1

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        size = (term_counts.get(node, 1) / max_count) * max_node_size
        node_sizes.append(size)

        node_colors.append(get_color(node))

        hover_text.append(
            f"GO ID: {node}<br>"
            f"Genes: {term_counts.get(node, 0)}<br>"
            f"p-value: {term_pvalues.get(node, 'N/A')}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color="black"),
        ),
        textposition="top center"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="GO Term Interaction Network",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def plot_go_interaction_network_html(
    gene2terms: Dict[str, List[str]],
    term_pvalues: Dict[str, float],
    gaf_path: PathLike,
    obo_path: PathLike,
    *,
    options: GoNetworkOptions = GoNetworkOptions(),
    save_html_to: Optional[PathLike] = None,
    return_fig: bool = False,
    return_html: bool = False,
):

    _validate_inputs(gene2terms, term_pvalues)

    # Load GO
    go3.load_go_terms(str(obo_path))
    annotations = go3.load_gaf(str(gaf_path))
    counter = go3.build_term_counter(annotations)

    # Count genes per term
    term_counts = _count_terms(gene2terms)

    # Filter terms
    term_list = [
        t for t, count in term_counts.items()
        if count >= options.min_genes_per_term
    ]

    if not term_list:
        raise ValueError("No GO terms passed filtering.")

    # Build graph
    G = _build_similarity_graph(
        term_list,
        counter,
        options.similarity_threshold,
        options.verbose,
    )

    if G.number_of_nodes() == 0:
        _log("No nodes available to build figure.", options.verbose)
        return None

    # Layout
    pos = _compute_layout(G, options.layout, options.random_state)

    # Plot
    fig = _build_plotly_figure(
        G,
        pos,
        term_counts,
        term_pvalues,
        options.max_node_size,
    )

    html = None

    if save_html_to or return_html:
        html = fig.to_html(include_plotlyjs="cdn", full_html=True)

        if save_html_to:
            p = Path(save_html_to)
            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(html, encoding="utf-8")
            _log(f"[GO] HTML saved at: {p}", options.verbose)

    if return_fig and return_html:
        return fig, html
    if return_fig:
        return fig
    if return_html:
        return html
    return None