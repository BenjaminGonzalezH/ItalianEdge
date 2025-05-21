import networkx as nx
import matplotlib.pyplot as plt
from pygosemsim.similarity import wang
from pygosemsim import graph
from goatools.obo_parser import GODag
from itertools import combinations
import numpy as np

def plot_go_interaction_network_py(gene2terms: dict[str, list[str]], 
                                   similarity_threshold: float = 0.7, 
                                   save_path: str = None,
                                   figsize=(12, 10), 
                                   title="GO Term Interaction Network"):
    """
    Construye y visualiza una red de interacción entre términos GO basada en similitud Wang.

    Parameters:
    - gene2terms: Diccionario {EntrezID: [GO terms]}
    - similarity_threshold: Umbral mínimo de similitud para crear una arista.
    - save_path: Ruta para guardar el gráfico (si None, solo se muestra).
    - weight_factor: Tuple con los pesos (is_a, part_of) para Wang.
    - figsize: Tamaño del gráfico.
    - title: Título del gráfico.
    """

    # 1. Cargar el grafo GO
    Gograph = graph.from_resource("go")  # o graph.from_obo("go-basic.obo")

    # 2. Obtener todos los términos únicos presentes
    all_terms = set(t for terms in gene2terms.values() for t in terms if t in Gograph)
    term_list = sorted(all_terms)

    # 3. Calcular matriz de similitud entre términos
    similarity_matrix = np.zeros((len(term_list), len(term_list)))

    for i, j in combinations(range(len(term_list)), 2):
        sim = wang(Gograph, term_list[i], term_list[j])
        if sim >= similarity_threshold:
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    # 4. Crear grafo con networkx
    G = nx.Graph()
    G.add_nodes_from(term_list)

    for i in range(len(term_list)):
        for j in range(i+1, len(term_list)):
            if similarity_matrix[i, j] >= similarity_threshold:
                G.add_edge(term_list[i], term_list[j], weight=similarity_matrix[i, j])

    # 5. Visualizar
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42, k=0.5)

    nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(title)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafo guardado en {save_path}")
    else:
        plt.show()