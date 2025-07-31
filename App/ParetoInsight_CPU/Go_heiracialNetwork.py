# Libraries
import networkx as nx                                   # Networks structures.
import plotly.graph_objects as go                       # plot figure.
import os                                               # Syscalls.
import urllib.request                                   # Query requests managment.
from goatools.obo_parser import GODag                   # GoDag managment.
import matplotlib.colors as mcolors                     # Color scale.
import numpy as np                                      # Para cálculos matemáticos
import json                                             # Para pasar datos a JavaScript

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

def separate_overlapping_nodes(pos, box_width=0.12, box_height=0.06, min_separation=0.03):
    """
    Ajusta posiciones para evitar superposición horizontal y vertical entre nodos.
    
    Parameters:
        pos: Diccionario con posiciones {node: (x, y)}
        box_width: Ancho de las cajas
        box_height: Alto de las cajas
        min_separation: Separación mínima entre cajas
        
    Returns:
        pos_adjusted: Posiciones ajustadas
    """
    pos_adjusted = pos.copy()

    # Agrupa nodos por nivel (según Y redondeado por altura)
    levels = {}
    for node, (x, y) in pos.items():
        level_key = round(y / box_height) * box_height
        if level_key not in levels:
            levels[level_key] = []
        levels[level_key].append((node, x, y))

    # Ordena niveles verticalmente
    sorted_levels = sorted(levels.items())

    # Ajuste horizontal por nivel
    for level_y, level_nodes in sorted_levels:
        if len(level_nodes) <= 1:
            continue

        level_nodes.sort(key=lambda item: item[1])  # ordena por x
        total_width_needed = len(level_nodes) * (box_width * 2 + min_separation)
        min_x = min(x for _, x, _ in level_nodes)
        max_x = max(x for _, x, _ in level_nodes)
        current_range = max_x - min_x

        if current_range < total_width_needed:
            center_x = (min_x + max_x) / 2
            start_x = center_x - (total_width_needed / 2)
            for i, (node, _, _) in enumerate(level_nodes):
                new_x = start_x + i * (box_width * 2 + min_separation) + box_width
                pos_adjusted[node] = (new_x, level_y)
        else:
            for i in range(len(level_nodes) - 1):
                node_a, x_a, y_a = level_nodes[i]
                node_b, x_b, y_b = level_nodes[i + 1]
                min_dist = box_width * 2 + min_separation
                if x_b - x_a < min_dist:
                    adjustment = min_dist - (x_b - x_a)
                    new_x = x_b + adjustment
                    pos_adjusted[node_b] = (new_x, level_y)
                    level_nodes[i + 1] = (node_b, new_x, level_y)

    # Ajuste vertical entre niveles
    min_vertical_gap = box_height + min_separation
    new_y_map = {}
    current_y = 0.0
    for level_y, level_nodes in sorted_levels:
        for node, x, y in level_nodes:
            pos_adjusted[node] = (pos_adjusted[node][0], current_y)
        new_y_map[level_y] = current_y
        current_y += min_vertical_gap

    return pos_adjusted

def get_node_paths(graph, node):
    """Obtiene todos los ancestros y descendientes de un nodo"""
    ancestors = set()
    descendants = set()
    
    # Obtener ancestros (hacia arriba en la jerarquía)
    stack = [node]
    while stack:
        current = stack.pop()
        for parent in graph.predecessors(current):
            if parent not in ancestors:
                ancestors.add(parent)
                stack.append(parent)
    
    # Obtener descendientes (hacia abajo en la jerarquía)
    stack = [node]
    while stack:
        current = stack.pop()
        for child in graph.successors(current):
            if child not in descendants:
                descendants.add(child)
                stack.append(child)
    
    return ancestors, descendants

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
    
    # Layout inicial
    pos = nx.multipartite_layout(G, subset_key="level", align='horizontal')
    pos = {node: (x, -y) for node, (x, y) in pos.items()}
    
    # Separar nodos superpuestos
    pos = separate_overlapping_nodes(pos, box_width=0.001, min_separation=0.9)

    # Preparar datos para JavaScript (información de conectividad)
    graph_data = {}
    for node in G.nodes():
        ancestors, descendants = get_node_paths(G, node)
        graph_data[node] = {
            'ancestors': list(ancestors),
            'descendants': list(descendants),
            'position': pos[node]
        }

    ######################################################################################################## Colors by p-value.
    cmap = mcolors.LinearSegmentedColormap.from_list("white_orange_red", ["#ffffff", "#ffa500", "#ff0000"])
    # Escala basada en -log10(p-value)
    log_pvals = [-np.log10(p + 1e-10) for p in term_pvalues.values()]
    norm = mcolors.Normalize(vmin=min(log_pvals), vmax=max(log_pvals))

    def get_color(term):
        pval = term_pvalues.get(term, 0.05)
        logp = -np.log10(pval + 1e-10)
        return mcolors.to_hex(cmap(norm(logp)))

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
                standoff=5,
                name=f"edge_{edge[0]}_{edge[1]}"  # Nombre para poder modificar después
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

    shapes = []
    hover_x = []
    hover_y = []
    hover_text = []

    for node in G.nodes():
        x, y = pos[node]
        w, h = 0.12, 0.06  # Cajas más pequeñas

        term_id = node
        term_name = G.nodes[node]['go_name']
        pval = term_pvalues.get(term_id, 1.0)
        associated_genes = gene2terms.get(term_id, [])
        n_genes = len(associated_genes)

        # RECTÁNGULO (solo color, sin texto)
        shapes.append(dict(
            type="rect",
            x0=x - w, x1=x + w,
            y0=y - h, y1=y + h,
            line=dict(color="black", width=1),
            fillcolor=get_color(term_id),
            layer="below",
            name=f"rect_{term_id}"  # Nombre para poder modificar después
        ))

        # TEXTO PARA TOOLTIP (hover) - información completa
        hover_x.append(x)
        hover_y.append(y)
        hover_text.append(f"<b>{term_name}</b><br>GO: {term_id}<br>p-value: {pval:.2e}<br>Genes: {n_genes}<br><i>Click to highlight path</i>")

    ######################################################################################################## Figure con colorbar
    fig = go.Figure()
    
    colorbar_trace = go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale=[[0, "#ffffff"], [0.5, "#ffa500"], [1, "#ff0000"]],
            showscale=True,
            cmin=min(log_pvals),
            cmax=max(log_pvals),
            colorbar=dict(
                title=dict(text="p-value)"),
                tickmode="linear",
                tick0=min(log_pvals),
                dtick=(max(log_pvals) - min(log_pvals)) / 5,
                tickformat=".1f",
                x=1.02,
                xanchor="left",
                len=0.8
            )
        ),
        hoverinfo='skip',
        showlegend=False
    )
    fig.add_trace(colorbar_trace)
    
    fig.update_layout(
        shapes=shapes,
        title="GO Term Hierarchical Tree - Click nodes to highlight paths",
        showlegend=False,
        margin=dict(b=20, l=20, r=120, t=60),  # Más margen a la derecha para la colorbar
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    fig.update_layout(annotations=fig.layout.annotations + tuple(arrow_annotations))

    # Agregar puntos clickeables para resaltar caminos
    node_trace = go.Scatter(
        x=hover_x,
        y=hover_y,
        mode='markers',
        marker=dict(size=25, color='rgba(0,0,0,0)'),  # Área de click más grande
        hoverinfo='text',
        hovertext=hover_text,
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font=dict(size=12)
        ),
        showlegend=False,
        customdata=[term for term in G.nodes()],  # IDs de los términos para el click
    )
    fig.add_trace(node_trace)

    # Definir ruta por defecto
    if save_path is None:
        save_path = "go_hierarchy.html"

    # Crear directorio si no existe
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Guardar la figura base
    fig.write_html(save_path)
    
    # Leer el archivo HTML y agregar JavaScript personalizado
    with open(save_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # JavaScript para manejar clicks y resaltar caminos
    custom_js = f'''
    <script>
    var graphData = {json.dumps(graph_data)};
    var originalShapes = null;
    var originalAnnotations = null;
    
    document.addEventListener('DOMContentLoaded', function() {{
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        
        gd.on('plotly_click', function(eventData) {{
            if (eventData.points.length > 0) {{
                var clickedTerm = eventData.points[0].customdata;
                highlightPath(clickedTerm, gd);
            }}
        }});
        
        // Doble click para resetear
        gd.on('plotly_doubleclick', function() {{
            resetHighlight(gd);
        }});
    }});
    
    function highlightPath(clickedTerm, gd) {{
        // Guardar estado original si no se ha guardado
        if (originalShapes === null) {{
            originalShapes = JSON.parse(JSON.stringify(gd.layout.shapes));
        }}
        if (originalAnnotations === null) {{
            originalAnnotations = JSON.parse(JSON.stringify(gd.layout.annotations));
        }}
        
        var pathNodes = new Set([clickedTerm]);
        
        // Agregar ancestros y descendientes
        if (graphData[clickedTerm]) {{
            graphData[clickedTerm].ancestors.forEach(function(ancestor) {{
                pathNodes.add(ancestor);
            }});
            graphData[clickedTerm].descendants.forEach(function(descendant) {{
                pathNodes.add(descendant);
            }});
        }}
        
        // Modificar opacidad de las formas
        var newShapes = gd.layout.shapes.map(function(shape) {{
            if (shape.name && shape.name.startsWith('rect_')) {{
                var termId = shape.name.replace('rect_', '');
                if (pathNodes.has(termId)) {{
                    // Resaltar nodos del camino
                    return {{
                        ...shape,
                        line: {{...shape.line, width: 3, color: 'red'}}
                    }};
                }} else {{
                    // Atenuar otros nodos
                    return {{
                        ...shape,
                        opacity: 0.3
                    }};
                }}
            }}
            return shape;
        }});
        
        // Modificar opacidad de las flechas
        var newAnnotations = gd.layout.annotations.map(function(annotation) {{
            if (annotation.name && annotation.name.startsWith('edge_')) {{
                var edgeParts = annotation.name.replace('edge_', '').split('_');
                var source = edgeParts[0];
                var target = edgeParts[1];
                
                if (pathNodes.has(source) && pathNodes.has(target)) {{
                    // Resaltar flechas del camino
                    return {{
                        ...annotation,
                        arrowcolor: 'blue',
                        arrowwidth: 3
                    }};
                }} else {{
                    // Atenuar otras flechas
                    return {{
                        ...annotation,
                        opacity: 0.3
                    }};
                }}
            }}
            return annotation;
        }});
        
        Plotly.relayout(gd, {{
            shapes: newShapes,
            annotations: newAnnotations
        }});
    }}
    
    function resetHighlight(gd) {{
        if (originalShapes && originalAnnotations) {{
            Plotly.relayout(gd, {{
                shapes: originalShapes,
                annotations: originalAnnotations
            }});
        }}
    }}
    </script>
    '''
    
    # Insertar JavaScript antes del cierre del body
    html_content = html_content.replace('</body>', custom_js + '</body>')
    
    # Escribir el archivo modificado
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive tree with path highlighting saved at: {save_path}")
    print("Click on nodes to highlight their hierarchical paths. Double-click to reset.")

    return fig