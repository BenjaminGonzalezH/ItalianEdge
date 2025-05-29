# Libraries.
import networkx as nx                                   # Networks structures.
import plotly.graph_objects as go                       # plot figure.
from pygosemsim.similarity import wang                  # Wang index function.
from pygosemsim import graph                            # Create GoDag from source.
from pygosemsim import download                         # Obtain Go obo file.
from itertools import combinations                      # Combinations
import matplotlib.colors as mcolors                     # Color scale.
import os                                               # Syscalls.

def plot_go_interaction_network_html(gene2terms: dict[str, list[str]],
                                     term_pvalues: dict[str, float],
                                     similarity_threshold: float = 0.7,
                                     download_f: bool = True,
                                     save_path: str | None = None):
    """
    plot_go_interaction_network_html(function): Build and generate a HTML image of a interactive
    netwrok of GO terms.
    
    Parameters:
        - gene2terms : Dictionary {GO Term: Genes associated}.
        - term_pvalues : Dictionary {GO Term: p-value}.
        - similarity_threshold : Similarity limit for connection in network.
        
    Returns:
        - fig: Figure created with plotly.
    """
    # Download managment.
    if download_f:
        try:
            download.clear()
            download.obo("go")
        except Exception as e:
            raise RuntimeError(f"Error in download GO OBO: {e}")
    # Load go data.
    go_graph = graph.from_resource("go")
    
    # Filter no found go terms in godag from file.
    all_terms = set()
    for terms in gene2terms.values():
        all_terms.update(terms)
    term_list = [t for t in all_terms if t in go_graph]
    
    # Contar cuántos genes están asociados a cada término
    term_counts = {}
    for term in term_list:
        count = sum(1 for genes in gene2terms.values() if term in genes)
        term_counts[term] = count
    
    ######################################################################################################## Construct graph usgin wang similarity.
    G = nx.Graph()
    for i, j in combinations(term_list, 2):
        try:
            sim = wang(go_graph, i, j)
            if sim >= similarity_threshold:
                G.add_edge(i, j, weight=sim)
        except Exception as e:
            print(f"Error calculating similarity between {i} and {j}: {e}")
            continue
    
    # Check empty graph.
    if len(G.nodes()) == 0:
        print("There is no nodes to create the figure.")
        return None
    
    # Positions of nodes.
    pos = nx.spring_layout(G, seed=42)
    
    # 5. Generar color de nodos según p-value
    # Asegurarse de que todos los términos tienen un p-valor
    for term in G.nodes():
        if term not in term_pvalues:
            term_pvalues[term] = 0.05  # Valor por defecto
    
    ######################################################################################################## Colors by p-value.
    cmap = mcolors.LinearSegmentedColormap.from_list("pvalue", ["red", "yellow", "green"])
    pvalues = [term_pvalues.get(term, 0.05) for term in G.nodes()]
    if pvalues:
        norm = mcolors.Normalize(vmin=min(pvalues), vmax=max(pvalues))
    else:
        norm = mcolors.Normalize(vmin=0, vmax=1)
    
    def get_color(term):
        return mcolors.to_hex(cmap(norm(term_pvalues.get(term, 0.05))))
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_widths = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_widths.append(edge[2]["weight"] * 2)
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="gray"),
        hoverinfo="none",
        mode="lines"
    )
    
    ######################################################################################################## Create nodes, size and text of every node.
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []
    hover_texts = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        # size using genes associated.
        size = term_counts.get(node, 1) * 3
        node_sizes.append(size)
        
        # p-value color.
        node_colors.append(get_color(node))
        
        # Information hover.
        hover_text = f"ID: {node}<br>Genes: {term_counts.get(node, 0)}<br>p-value: {term_pvalues.get(node, 'N/A')}"
        hover_texts.append(hover_text)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text=node_text,
        mode="markers+text",
        hovertext=hover_texts,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color="black")
        ),
        textposition="top center"
    )
    
    ######################################################################################################## Create figure with plotly.
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="GO Term Interaction Network",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
        )
    )
    
    # Save HTML.
    if save_path is None:
        save_path = "go_network.html"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    try:
        fig.write_html(save_path)
        print(f"Network saved at: {save_path}")
    except Exception as e:
        print(f"Unexpected error creating network: {e}")
    
    return fig