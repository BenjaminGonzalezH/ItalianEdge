# Libraries
import networkx as nx
import plotly.graph_objects as go
import os
import urllib.request
from goatools.obo_parser import GODag
import matplotlib.colors as mcolors


def download_go_obo(obo_path="go.obo", force_download=False):
    url = "http://purl.obolibrary.org/obo/go/go.obo"
    if os.path.exists(obo_path) and not force_download:
        print(f"{obo_path} exist, no download performed.")
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, obo_path)
    print("Completed")
    return obo_path


def plot_go_hierarchy_html(gene2terms: dict[str, list[str]],
                           term_pvalues: dict[str, float],
                           max_nodes: int = 100,
                           download_f: bool = True):
    """
    Genera un árbol jerárquico de términos GO en HTML interactivo.

    Parameters:
    -----------
    gene2terms : dict[str, list[str]]
        Diccionario con {GO Term: Genes asociados}
    term_pvalues : dict[str, float]
        Diccionario con {GO Term: p-value de cada término}
    ontology : str, default="BP"
        Aspecto GO a representar ('BP', 'MF', 'CC')
    max_nodes : int, default=50
        Máximo número de términos GO en el árbol
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figura de Plotly con el árbol GO jerárquico.
    """
    
    # 1. Cargar ontología GO
    if download_f:
        download_go_obo()
    go_dag = GODag("go.obo")

    # 2. Filtrar términos según el aspecto GO elegido
    all_terms = set(term for terms in gene2terms.values() for term in terms if term in go_dag)
    go_terms = [t for t in all_terms if go_dag[t]]

    # Ordenar y limitar términos GO por p-value
    go_terms = sorted(go_terms, key=lambda x: term_pvalues.get(x, 1.0))[:max_nodes]

    # 3. Construcción del árbol GO usando `DiGraph`
    G = nx.DiGraph()
    for go_term in go_terms:
        G.add_node(go_term, go_name=go_dag[go_term].name, 
                p_value=term_pvalues.get(go_term, 1.0),
                level=go_dag[go_term].depth)  # Asigna la profundidad

        for parent_term in go_dag[go_term].parents:
            parent_id = parent_term.id  # Extraer ID string
            if parent_id in go_terms:
                G.add_edge(parent_id, go_term)

    # 4. Obtener posiciones jerárquicas con `networkx`
    pos = nx.multipartite_layout(G, subset_key="level")

    # 5. Definir colores según p-value
    cmap = mcolors.LinearSegmentedColormap.from_list("pvalue", ["red", "yellow", "green"])
    norm = mcolors.Normalize(vmin=min(term_pvalues.values()), vmax=max(term_pvalues.values()))

    def get_color(term):
        return mcolors.to_hex(cmap(norm(term_pvalues.get(term, 0.05))))

    # 6. Crear aristas
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="gray"), hoverinfo="none")

    # 7. Crear nodos con tamaños y colores personalizados
    node_x, node_y, node_text, node_sizes, node_colors = [], [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{G.nodes[node]['go_name']} ({node})")
        node_sizes.append(15)
        node_colors.append(get_color(node))

    node_trace = go.Scatter(x=node_x, y=node_y, text=node_text, mode="markers+text",
                            hoverinfo="text", marker=dict(size=node_sizes, color=node_colors, line=dict(width=1)))

    # Preparar menú desplegable para resaltar nodos
    terms = list(G.nodes())
    base_colors = node_colors
    highlight_color = "cyan"

    buttons = []
    for i, term in enumerate(terms):
        colors = [highlight_color if j == i else base_colors[j] for j in range(len(terms))]
        sizes = [node_sizes[j]*1.5 if j == i else node_sizes[j] for j in range(len(terms))]
        buttons.append(dict(
            label=term,
            method="restyle",
            args=[{
                "marker.color": [colors],
                "marker.size": [sizes],
            }, [1]]  # índice 1 es node_trace
        ))

    # Botón para deseleccionar
    buttons.insert(0, dict(
        label="Ninguno",
        method="restyle",
        args=[{
            "marker.color": [base_colors],
            "marker.size": [node_sizes],
        }, [1]]
    ))

    fig = go.Figure(data=[edge_trace, node_trace])

    # Actualizar layout para incluir menú
    fig.update_layout(
        title="GO Term Hierarchical Tree",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.1,
            y=1.15,
            xanchor="left",
            yanchor="top"
        )]
    )

    fig.write_html("go_hierarchy.html")
    print("Árbol GO guardado como go_hierarchy.html")

    return fig