import networkx as nx
import plotly.graph_objects as go
from pygosemsim.similarity import wang
from pygosemsim import graph
from pygosemsim import download
from itertools import combinations
import matplotlib.colors as mcolors

def plot_go_interaction_network_html(gene2terms: dict[str, list[str]],
                                     term_pvalues: dict[str, float],
                                     similarity_threshold: float = 0.7,
                                     download_f: bool = True):
    """
    Construye y genera una versión HTML interactiva de la red de interacción entre términos GO.
    
    Parameters:
    -----------
    gene2terms : dict[str, list[str]]
        Diccionario donde la clave es el término GO y el valor es una lista de genes asociados
    term_pvalues : dict[str, float]
        Diccionario donde la clave es el término GO y el valor es su p-value
    similarity_threshold : float, default=0.7
        Umbral de similitud para conectar términos GO en la red
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figura de Plotly con la red de interacción
    """
    # Download managment.
    if download_f:
        try:
            download.clear()
            download.obo("go")
        except Exception as e:
            raise RuntimeError(f"Error in download GO OBO: {e}")

    # 1. Cargar el grafo GO
    go_graph = graph.from_resource("go")
    
    # 2. Obtener términos únicos y calcular frecuencia
    # Corregido: term_counts debe contar genes asociados a cada término
    all_terms = set()
    for terms in gene2terms.values():
        all_terms.update(terms)
    
    # Filtrar términos que existen en el grafo GO
    term_list = [t for t in all_terms if t in go_graph]
    
    # Contar cuántos genes están asociados a cada término
    term_counts = {}
    for term in term_list:
        count = sum(1 for genes in gene2terms.values() if term in genes)
        term_counts[term] = count
    
    # 3. Construir grafo con pesos
    G = nx.Graph()
    for i, j in combinations(term_list, 2):
        try:
            sim = wang(go_graph, i, j)
            if sim >= similarity_threshold:
                G.add_edge(i, j, weight=sim)
        except Exception as e:
            print(f"Error calculating similarity between {i} and {j}: {e}")
            continue
    
    # Si el grafo está vacío, no podemos crear la visualización
    if len(G.nodes()) == 0:
        print("No hay suficientes términos GO con similitud por encima del umbral")
        return None
    
    # 4. Obtener posiciones con layout de NetworkX
    pos = nx.spring_layout(G, seed=42)
    
    # 5. Generar color de nodos según p-value
    # Asegurarse de que todos los términos tienen un p-valor
    for term in G.nodes():
        if term not in term_pvalues:
            term_pvalues[term] = 0.05  # Valor por defecto
    
    # Crear mapa de colores
    cmap = mcolors.LinearSegmentedColormap.from_list("pvalue", ["red", "yellow", "green"])
    pvalues = [term_pvalues.get(term, 0.05) for term in G.nodes()]
    if pvalues:  # Comprobar que hay p-valores
        norm = mcolors.Normalize(vmin=min(pvalues), vmax=max(pvalues))
    else:
        norm = mcolors.Normalize(vmin=0, vmax=1)
    
    def get_color(term):
        return mcolors.to_hex(cmap(norm(term_pvalues.get(term, 0.05))))
    
    # 6. Crear aristas
    edge_x = []
    edge_y = []
    edge_widths = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_widths.append(edge[2]["weight"] * 2)  # Grosor según similitud
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="gray"),
        hoverinfo="none",
        mode="lines"
    )
    
    # 7. Crear nodos con tamaño, color y etiquetas
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
        
        # Tamaño según genes asociados
        size = term_counts.get(node, 1) * 3
        node_sizes.append(size)
        
        # Color por p-value
        node_colors.append(get_color(node))
        
        # Texto de hover con información
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
    
    # 8. Crear figura interactiva con Plotly
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
    
    # 9. Guardar como archivo HTML
    try:
        fig.write_html("go_network.html")
        print("Archivo HTML guardado: go_network.html")
    except Exception as e:
        print(f"Error al guardar el archivo HTML: {e}")
    
    return fig