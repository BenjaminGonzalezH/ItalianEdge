"""
Hypergraph-based consensus clustering.

Features:
- CSPA with spectral or agglomerative partitioning.
- MCLA compatible with Jaccard, Rand and Adjusted Rand.
- Integrated visualization utilities.
"""

# ──────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────
import numpy as np
import networkx as nx
import plotly.graph_objects as go

from typing import Literal
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.manifold import spectral_embedding

from .ConsensusMatrix import consensus_matrix
from .RandValues import (
    rand_index_clusters,
    adjusted_rand_index_clusters
)
from .JaccardValues import jaccard_index_clusters


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────

def _validate_labels_matrix(labels_matrix: np.ndarray) -> None:
    if not isinstance(labels_matrix, np.ndarray):
        raise TypeError("labels_matrix must be numpy.ndarray.")
    if labels_matrix.ndim != 2:
        raise ValueError("labels_matrix must be 2D.")
    if labels_matrix.shape[0] == 0:
        raise ValueError("Empty labels matrix.")
    if labels_matrix.shape[1] < 2:
        raise ValueError("At least 2 elements required.")


# =============================================================
# CSPA
# =============================================================

def cspa_consensus(
    labels_matrix: np.ndarray,
    num_clusters: int,
    method: Literal["spectral", "agglomerative"] = "spectral",
    linkage: str = "average"
) -> np.ndarray:
    """
    CSPA consensus clustering.

    method:
        - "spectral"
        - "agglomerative"
    """

    _validate_labels_matrix(labels_matrix)

    coincidence, distance = consensus_matrix(labels_matrix)

    if method == "spectral":

        model = SpectralClustering(
            n_clusters=num_clusters,
            affinity="precomputed",
            assign_labels="discretize",
            random_state=0
        )
        return model.fit_predict(coincidence)

    elif method == "agglomerative":

        model = AgglomerativeClustering(
            n_clusters=num_clusters,
            metric="precomputed",
            linkage=linkage
        )
        return model.fit_predict(distance)

    else:
        raise ValueError("method must be 'spectral' or 'agglomerative'")


# =============================================================
# MCLA
# =============================================================

def mcla_consensus(
    labels_matrix: np.ndarray,
    num_clusters: int,
    metric: Literal["jaccard", "rand", "adjusted_rand"] = "jaccard",
    linkage: str = "average"
) -> np.ndarray:

    _validate_labels_matrix(labels_matrix)

    n_solutions, n_elements = labels_matrix.shape

    # Build cluster vectors
    cluster_vectors = []
    cluster_sets = []

    for solution in labels_matrix:
        labels = np.unique(solution)
        for label in labels:
            vec = (solution == label).astype(np.int8)
            cluster_vectors.append(vec)
            cluster_sets.append(set(np.where(solution == label)[0]))

    cluster_vectors = np.array(cluster_vectors)
    n_clusters_total = cluster_vectors.shape[0]

    if n_clusters_total < num_clusters:
        raise ValueError("Not enough clusters to build meta-clusters.")

    # Similarity matrix between clusters
    sim = np.zeros((n_clusters_total, n_clusters_total))

    for i in range(n_clusters_total):
        for j in range(i, n_clusters_total):

            if metric == "jaccard":
                value = len(cluster_sets[i] & cluster_sets[j]) / \
                        len(cluster_sets[i] | cluster_sets[j]) \
                        if len(cluster_sets[i] | cluster_sets[j]) > 0 else 0.0

            elif metric == "rand":
                value = rand_index_clusters(
                    [cluster_sets[i]],
                    [cluster_sets[j]]
                )[0, 0]

            elif metric == "adjusted_rand":
                value = adjusted_rand_index_clusters(
                    [cluster_sets[i]],
                    [cluster_sets[j]]
                )[0, 0]

            else:
                raise ValueError("Invalid metric.")

            sim[i, j] = sim[j, i] = value

    distance = 1.0 - sim

    meta_model = AgglomerativeClustering(
        n_clusters=num_clusters,
        metric="precomputed",
        linkage=linkage
    )

    meta_labels = meta_model.fit_predict(distance)

    # Voting assignment
    consensus_labels = np.zeros(n_elements, dtype=int)

    for element in range(n_elements):

        votes = np.zeros(num_clusters, dtype=int)

        for cluster_idx, meta_label in enumerate(meta_labels):
            if cluster_vectors[cluster_idx, element] == 1:
                votes[meta_label] += 1

        consensus_labels[element] = votes.argmax()

    return consensus_labels


# =============================================================
# VISUALIZATION
# =============================================================

def plot_cspa_graph(coincidence_matrix, threshold=0.3):

    G = nx.Graph()
    n = coincidence_matrix.shape[0]

    for i in range(n):
        G.add_node(i)

    for i in range(n):
        for j in range(i + 1, n):
            if coincidence_matrix[i, j] >= threshold:
                G.add_edge(i, j, weight=coincidence_matrix[i, j])

    pos = nx.spring_layout(G, seed=0)

    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1),
        hoverinfo="none"
    ))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(size=10),
        text=[str(i) for i in G.nodes()],
        hoverinfo="text"
    ))

    fig.update_layout(title="CSPA Co-association Graph")
    return fig


def plot_cspa_embedding(coincidence_matrix, labels):

    embedding = spectral_embedding(
        coincidence_matrix,
        n_components=2,
        random_state=0
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        mode="markers",
        marker=dict(
            size=10,
            color=labels,
            colorscale="Viridis"
        ),
        text=[str(i) for i in range(len(labels))]
    ))

    fig.update_layout(title="Spectral Embedding")
    return fig


def plot_mcla_metagraph(similarity_matrix):

    G = nx.from_numpy_array(similarity_matrix)

    pos = nx.spring_layout(G, seed=0)

    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1),
        hoverinfo="none"
    ))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(size=12),
        text=[str(i) for i in G.nodes()],
        hoverinfo="text"
    ))

    fig.update_layout(title="MCLA Meta-Cluster Graph")
    return fig