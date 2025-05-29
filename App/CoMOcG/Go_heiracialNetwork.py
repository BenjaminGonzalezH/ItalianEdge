# Libraries
import networkx as nx                                   # Networks structures.
import plotly.graph_objects as go                       # plot figure.
import os                                               # Syscalls.
import urllib.request                                   # Query requests managment.
from goatools.obo_parser import GODag                   # GoDag managment.
import matplotlib.colors as mcolors                     # Color scale.

######### Functions #########

"""
This block contains all main functions.
"""

def download_go_obo(obo_path="go.obo", force_download=False):
    """
    download_go_obo(function): Download go.obo file just for use with goatools. No related
    to pygosemsim.

    Parameters:
    obo_path: Just the file associated with common url of this files.
    force_download: Overwrite original file.
    """
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
                           download_f: bool = True,
                           save_path: str | None = None):
    """
    plot_go_hierarchy_html(function): Create a Go DAG representing heriacial relation among terms.

    Parameters:
        - gene2terms: Dictionary {GO Term: Genes associated}
        - term_pvalues: Dictionary {GO Term: p-value}
        - ontology: Go aspect ('BP', 'MF', 'CC')
        - max_nodes: Maximun nodes allowed in tree.
        - save_path: Location in your computer to allocate the html.
        
    Returns:
        - fig: Figure created with plotly.
    """
    # Download ontology if it is needed.
    if download_f:
        download_go_obo(force_download=True)
    # Load go data.
    go_dag = GODag("go.obo")

    # Filter no found go terms in godag from file.
    all_terms = set(term for terms in gene2terms.values() for term in terms if term in go_dag)
    go_terms = [t for t in all_terms if go_dag[t]]

    # Sort and cut terms by p-value.
    go_terms = sorted(go_terms, key=lambda x: term_pvalues.get(x, 1.0))[:max_nodes]

    # Tree build using `DiGraph`
    G = nx.DiGraph()
    for go_term in go_terms:
        G.add_node(go_term, go_name=go_dag[go_term].name, 
                p_value=term_pvalues.get(go_term, 1.0),
                level=go_dag[go_term].depth)  # Depth assigment.

        for parent_term in go_dag[go_term].parents:
            parent_id = parent_term.id  # Obtain ID of parents to settle connection.
            if parent_id in go_terms:
                G.add_edge(parent_id, go_term)

    # Create levels for the figure.
    pos = nx.multipartite_layout(G, subset_key="level", align='horizontal')
    pos = {node: (x, -y) for node, (x, y) in pos.items()}

    ######################################################################################################## Colors by p-value.
    cmap = mcolors.LinearSegmentedColormap.from_list("white_orange_red", ["#ffffff", "#ffa500", "#ff0000"])
    norm = mcolors.Normalize(vmin=min(term_pvalues.values()), vmax=max(term_pvalues.values()))
    def get_color(term):
        return mcolors.to_hex(cmap(norm(term_pvalues.get(term, 0.05))))

    ######################################################################################################## Create edges.
    arrow_annotations = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        arrow_annotations.append(
            dict(
                ax=x0, ay=y0,
                x=x1, y=y1,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=3, arrowsize=1, arrowwidth=1.5, arrowcolor='gray',
                standoff=5
            )
        )

    ######################################################################################################## Create nodes
    node_x, node_y, node_text, node_sizes, node_colors = [], [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{G.nodes[node]['go_name']} ({node})")
        node_sizes.append(15)
        node_colors.append(get_color(node))

    node_trace = go.Scatter(x=node_x, y=node_y, text=node_text, mode="markers+text", textposition="middle center",
                            hoverinfo="text", marker=dict(size=node_sizes, symbol="square", color=node_colors, line=dict(width=1)))

    ######################################################################################################## Highlight nodes
    terms = list(G.nodes())
    base_colors = node_colors
    highlight_color = "cyan"

    buttons = []
    for i, term in enumerate(terms):
        colors = [highlight_color if j == i else base_colors[j] for j in range(len(terms))]
        sizes = [node_sizes[j]*1.5 if j == i else node_sizes[j] for j in range(len(terms))]
        label = f"{G.nodes[term]['go_name']} ({term})"
        buttons.append(dict(
            label=label,
            method="restyle",
            args=[{
                "marker.color": [colors],
                "marker.size": [sizes],
            }, [0]]
        ))

    buttons.insert(0, dict(
        label="None",
        method="restyle",
        args=[{
            "marker.color": [base_colors],
            "marker.size": [node_sizes],
        }, [0]]
    ))

    ######################################################################################################## Figure.
    fig = go.Figure(data=[node_trace])

    fig.update_layout(
        title="GO Term Hierarchical Tree",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=arrow_annotations,
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.049,
            y=1.01,
            xanchor="left",
            yanchor="top"
        )]
    )

    # Definir ruta por defecto
    if save_path is None:
        save_path = "go_hierarchy.html"

    # Crear directorio si no existe
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig.write_html(save_path)
    print(f"Tree saved at: {save_path}")

    return fig