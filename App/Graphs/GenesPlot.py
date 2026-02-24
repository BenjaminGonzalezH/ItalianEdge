from pathlib import Path
from typing import Optional, Sequence, Union, Literal
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from typing import Optional, Sequence, Union
import plotly.express as px
from sklearn.manifold import SpectralEmbedding


PathLike = Union[str, Path]


def plot_gene_similarity_network(
    similarity_matrix: np.ndarray,
    genes: Sequence[str],
    *,
    threshold: Optional[float] = 0.7,
    top_k: Optional[int] = None,
    layout: Literal["spring", "kamada_kawai"] = "spring",
    random_state: int = 42,
    edge_scale: float = 4.0,
    node_size: float = 10,
    save_html_to: Optional[PathLike] = None,
    return_fig: bool = False,
    return_html: bool = False,
):
    """
    Build and visualize an interactive gene-gene similarity network.

    Parameters
    ----------
    similarity_matrix : np.ndarray (NxN)
        Symmetric similarity matrix.
    genes : list[str]
        Gene identifiers.
    threshold : float, optional
        Add edge if similarity >= threshold.
    top_k : int, optional
        Alternatively connect each node to its top_k most similar genes.
        If provided, overrides threshold.
    layout : str
        "spring" or "kamada_kawai".
    edge_scale : float
        Scaling factor for edge thickness.
    node_size : float
        Node marker size.
    """

    # ──────────────────────────────────────
    # Validation
    # ──────────────────────────────────────
    if not isinstance(similarity_matrix, np.ndarray):
        raise TypeError("similarity_matrix must be numpy.ndarray")

    if similarity_matrix.ndim != 2:
        raise ValueError("similarity_matrix must be 2D")

    n = similarity_matrix.shape[0]

    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("similarity_matrix must be square")

    if len(genes) != n:
        raise ValueError("Number of genes must match matrix size")

    if not np.allclose(similarity_matrix, similarity_matrix.T, atol=1e-8):
        raise ValueError("similarity_matrix must be symmetric")

    if threshold is None and top_k is None:
        raise ValueError("Either threshold or top_k must be provided")

    # ──────────────────────────────────────
    # Build Graph
    # ──────────────────────────────────────
    G = nx.Graph()

    for g in genes:
        G.add_node(g)

    if top_k is not None:
        for i in range(n):
            idx = np.argsort(similarity_matrix[i])[::-1][1: top_k + 1]
            for j in idx:
                G.add_edge(
                    genes[i],
                    genes[j],
                    weight=float(similarity_matrix[i, j])
                )
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    G.add_edge(
                        genes[i],
                        genes[j],
                        weight=float(similarity_matrix[i, j])
                    )

    # ──────────────────────────────────────
    # Layout
    # ──────────────────────────────────────
    if layout == "spring":
        pos = nx.spring_layout(G, weight="weight", seed=random_state)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError("layout must be 'spring' or 'kamada_kawai'")

    # ──────────────────────────────────────
    # Edges
    # ──────────────────────────────────────
    edge_x = []
    edge_y = []
    edge_widths = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_widths.append(data["weight"] * edge_scale)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="gray"),
        hoverinfo="none"
    )

    # ──────────────────────────────────────
    # Nodes
    # ──────────────────────────────────────
    node_x = []
    node_y = []
    hover_text = []
    degrees = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        deg = G.degree(node)
        degrees.append(deg)

        hover_text.append(
            f"Gene: {node}<br>"
            f"Degree: {deg}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=node_size,
            color=degrees,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Degree")
        ),
        text=hover_text,
        hoverinfo="text"
    )

    # ──────────────────────────────────────
    # Figure
    # ──────────────────────────────────────
    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title="Gene-Gene Similarity Network",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    html = None

    if save_html_to or return_html:
        html = fig.to_html(include_plotlyjs="cdn", full_html=True)

        if save_html_to:
            p = Path(save_html_to)
            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(html, encoding="utf-8")

    if return_fig and return_html:
        return fig, html
    if return_fig:
        return fig
    if return_html:
        return html

    return None

def plot_gene_embedding_2d(
    similarity_matrix: np.ndarray,
    genes: Sequence[str],
    *,
    clusters: Optional[Sequence[int]] = None,
    random_state: int = 42,
    save_html_to: Optional[PathLike] = None,
    return_fig: bool = False,
    return_html: bool = False,
):
    """
    2D Spectral Embedding from a gene-gene similarity matrix.

    Parameters
    ----------
    similarity_matrix : np.ndarray (NxN)
        Symmetric similarity matrix.
    genes : list[str]
        Gene identifiers.
    clusters : optional list[int]
        Cluster labels for coloring.
    """

    # ──────────────────────────────────────
    # Validation
    # ──────────────────────────────────────
    if not isinstance(similarity_matrix, np.ndarray):
        raise TypeError("similarity_matrix must be numpy.ndarray")

    if similarity_matrix.ndim != 2:
        raise ValueError("similarity_matrix must be 2D")

    n = similarity_matrix.shape[0]

    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("similarity_matrix must be square")

    if len(genes) != n:
        raise ValueError("Number of genes must match matrix size")

    if not np.allclose(similarity_matrix, similarity_matrix.T, atol=1e-8):
        raise ValueError("similarity_matrix must be symmetric")

    if clusters is not None and len(clusters) != n:
        raise ValueError("clusters length must match number of genes")

    # ──────────────────────────────────────
    # Spectral Embedding
    # ──────────────────────────────────────
    embedding = SpectralEmbedding(
        n_components=2,
        affinity="precomputed",
        random_state=random_state
    )

    coords = embedding.fit_transform(similarity_matrix)

    # ──────────────────────────────────────
    # DataFrame for plotting
    # ──────────────────────────────────────
    import pandas as pd

    df = pd.DataFrame({
        "Gene": genes,
        "Dim1": coords[:, 0],
        "Dim2": coords[:, 1],
    })

    if clusters is not None:
        df["Cluster"] = clusters
        color_col = "Cluster"
    else:
        color_col = None

    # ──────────────────────────────────────
    # Plot
    # ──────────────────────────────────────
    fig = px.scatter(
        df,
        x="Dim1",
        y="Dim2",
        color=color_col,
        hover_data=["Gene"],
        title="Gene Similarity 2D Embedding"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )

    html = None

    if save_html_to or return_html:
        html = fig.to_html(include_plotlyjs="cdn", full_html=True)

        if save_html_to:
            p = Path(save_html_to)
            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(html, encoding="utf-8")

    if return_fig and return_html:
        return fig, html
    if return_fig:
        return fig
    if return_html:
        return html

    return None