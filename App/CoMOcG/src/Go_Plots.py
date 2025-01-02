import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib.patches import Circle
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from GoEnrischment import setup_r_environment

def map_genes_to_go_terms(df):
    """
    Relaciona genes con los términos GO en los que están involucrados.
    
    Parámetros:
    df (pd.DataFrame): DataFrame con dos columnas: 'GO_term' y 'Gene_IDs', donde 'Gene_IDs'
                        contiene los IDs de los genes asociados a cada término GO.

    Devuelve:
    dict: Diccionario donde las claves son los IDs de los genes y los valores son listas de términos GO
          asociados a cada gen.
    """
    gene_to_go = {}

    # Iterar sobre cada fila del DataFrame
    for _, row in df.iterrows():
        go_term = row['ID']
        gene_ids = row['geneID']
        gene_ids = [int(x) for x in gene_ids.split('/')]
        
        # Iterar sobre los genes asociados a cada término GO
        for gene in gene_ids:
            if gene not in gene_to_go:
                gene_to_go[gene] = []
            gene_to_go[gene].append(go_term)
    
    return gene_to_go

def generate_heatmap_from_genes(wang_similarity_df, genes_to_go):
    """
    Generate a heatmap showing the Wang similarity between genes based on GO terms.
    
    Parameters:
    dataframe (pd.DataFrame): Original dataframe with enriched GO terms.
    wang_similarity_df (pd.DataFrame): DataFrame with Wang similarity between GO terms.
    genes_to_go (dict): Dictionary mapping genes to GO terms.
    
    Returns:
    None: Displays the heatmap.
    """
    try:
        # Create a mapping of genes to GO terms
        genes_to_go = {}
        for _, row in wang_similarity_df.iterrows():
            go_term = row['ID']
            genes = row['geneID'].split('/')
            for gene in genes:
                if gene not in genes_to_go:
                    genes_to_go[gene] = []
                genes_to_go[gene].append(go_term)
        
        # Create a list of unique genes
        genes = list(genes_to_go.keys())
        num_genes = len(genes)

        # Initialize similarity matrix
        similarity_matrix = np.zeros((num_genes, num_genes))

        for i, gene_i in enumerate(genes):
            for j, gene_j in enumerate(genes):
                if i <= j:  # Fill upper triangle only
                    go_terms_i = genes_to_go[gene_i]
                    go_terms_j = genes_to_go[gene_j]
                    
                    # Calculate average Wang similarity between GO terms
                    similarities = []
                    for go_i in go_terms_i:
                        for go_j in go_terms_j:
                            sim = wang_similarity_df.loc[
                                (wang_similarity_df['ID'] == go_i) &
                                (wang_similarity_df['ID'] == go_j),
                                'wang_similarity'
                            ]
                            if not sim.empty:
                                similarities.append(sim.values[0])
                    
                    # Average similarity
                    similarity_matrix[i, j] = similarity_matrix[j, i] = (
                        np.mean(similarities) if similarities else 0
                    )

        # Create DataFrame for the similarity matrix
        similarity_df = pd.DataFrame(
            similarity_matrix, index=genes, columns=genes
        )

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_df, cmap='coolwarm', annot=False, square=True)
        plt.title("Wang Similarity Heatmap")
        plt.show()

    except Exception as e:
        print(f"Error generating heatmap: {e}")

def plot_gene_ratio(wang_similarity_df):
    """
    Plot the GeneRatio for GO terms.

    Parameters:
    wang_similarity_df (pd.DataFrame): DataFrame with enriched GO terms and associated data.

    Returns:
    None: Displays the plot.
    """
    try:
        plt.figure(figsize=(10, 6))
        sorted_df = wang_similarity_df.sort_values("GeneRatio", ascending=False)
        sns.barplot(
            y=sorted_df["Description"],
            x=sorted_df["GeneRatio"],
            palette="viridis"
        )
        plt.xlabel("Gene Ratio")
        plt.ylabel("GO Terms")
        plt.title("Gene Ratio for GO Terms")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting Gene Ratio: {e}")

def plot_qscore(wang_similarity_df):
    """
    Plot the qScore (negative log of qvalue) for GO terms.

    Parameters:
    wang_similarity_df (pd.DataFrame): DataFrame with enriched GO terms and associated data.

    Returns:
    None: Displays the plot.
    """
    try:
        wang_similarity_df['qScore'] = -np.log10(wang_similarity_df['qvalue'])
        sorted_df = wang_similarity_df.sort_values("qScore", ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            y=sorted_df["Description"],
            x=sorted_df["qScore"],
            palette="coolwarm"
        )
        plt.xlabel("-log10(qvalue)")
        plt.ylabel("GO Terms")
        plt.title("qScore for GO Terms")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting qScore: {e}")

def plot_go_interaction_network(wang_similarity_df, threshold=0.7):
    """
    Plot an interaction network for GO terms based on Wang similarity.

    Parameters:
    wang_similarity_df (pd.DataFrame): DataFrame with GO terms and Wang similarity.
    threshold (float): Minimum Wang similarity to create an edge.

    Returns:
    None: Displays the network plot.
    """
    try:
        # Create a graph
        G = nx.Graph()

        # Add nodes and edges
        for _, row in wang_similarity_df.iterrows():
            go_term = row['ID']
            G.add_node(go_term, label=row['Description'])
            
            # Check for high similarity between terms
            for _, other_row in wang_similarity_df.iterrows():
                if row['ID'] != other_row['ID']:
                    if row['wang_similarity'] >= threshold:
                        G.add_edge(row['ID'], other_row['ID'], weight=row['wang_similarity'])
        
        # Draw the network
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 12))
        nx.draw(
            G, pos, with_labels=True, node_size=700, node_color="skyblue",
            font_size=10, font_color="black", edge_color="gray"
        )
        plt.title("GO Interaction Network")
        plt.show()
    except Exception as e:
        print(f"Error plotting GO interaction network: {e}")

def create_emmaplot_network(df, similarity_threshold=0.7, min_fold_enrichment=2.0):
    """
    Creates an emmaplot-style network visualization of GO terms.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing GO terms analysis results with columns:
        'ID', 'Description', 'GeneRatio', 'p.adjust', 'Count', etc.
    similarity_threshold : float
        Minimum similarity score to create an edge between nodes
    min_fold_enrichment : float
        Minimum fold enrichment to include a term
    """
    # Create graph
    G = nx.Graph()
    
    # Filter significant terms
    significant_terms = df[df['FoldEnrichment'] >= min_fold_enrichment]
    
    # Add nodes
    for idx, row in significant_terms.iterrows():
        # Extract GeneRatio numerator and denominator
        gene_ratio = [float(x) for x in row['GeneRatio'].split('/')]
        ratio = gene_ratio[0] / gene_ratio[1]
        
        G.add_node(row['ID'],
                  description=row['Description'],
                  gene_count=row['Count'],
                  p_adjust=row['p.adjust'],
                  gene_ratio=ratio)
    
    # Add edges based on similarity
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            term1, term2 = nodes[i], nodes[j]
            similarity = df.loc[df['ID'] == term1, 'wang_similarity'].iloc[0]
            
            if isinstance(similarity, (int, float)) and similarity >= similarity_threshold:
                G.add_edge(term1, term2, weight=similarity)
    
    # Create figure
    plt.figure(figsize=(12, 12))
    
    # Calculate layout (using spring layout for better spacing)
    pos = nx.spring_layout(G, k=2, iterations=100)
    
    # Draw edges first
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, 
                          width=np.array(edge_weights) * 2,
                          alpha=0.3,
                          edge_color='gray')
    
    # Prepare node colors based on adjusted p-values
    p_values = np.array([G.nodes[node]['p_adjust'] for node in G.nodes()])
    p_values_log = -np.log10(p_values)
    
    # Prepare node sizes based on gene counts
    counts = np.array([G.nodes[node]['gene_count'] for node in G.nodes()])
    sizes = 100 + (counts - min(counts)) / (max(counts) - min(counts)) * 1000
    
    # Create color map
    cmap = plt.cm.RdYlBu_r
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                              norm=plt.Normalize(vmin=min(p_values_log),
                                               vmax=max(p_values_log)))
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos,
                                 node_size=sizes,
                                 node_color=p_values_log,
                                 cmap=cmap,
                                 alpha=0.7)
    
    # Add labels with wrapped text
    labels = {}
    for node in G.nodes():
        desc = G.nodes[node]['description']
        # Wrap description to multiple lines if too long
        if len(desc) > 20:
            words = desc.split()
            new_desc = ''
            line = ''
            for word in words:
                if len(line + ' ' + word) > 20:
                    new_desc += line + '\n'
                    line = word
                else:
                    line += ' ' + word if line else word
            new_desc += line
            labels[node] = new_desc
        else:
            labels[node] = desc
            
    nx.draw_networkx_labels(G, pos, labels,
                           font_size=8,
                           bbox=dict(facecolor='white',
                                   alpha=0.7,
                                   edgecolor='none',
                                   pad=0.5))
    
    # Add colorbar
    plt.colorbar(sm, ax=plt.gca(), label='-log10(p.adjust)')
    
    # Add legend for node sizes
    legend_elements = [
        Circle((0, 0), radius=np.sqrt(s/(100*np.pi)), 
               facecolor='gray', alpha=0.5,
               label=f'Count: {c}')
        for s, c in zip([min(sizes), max(sizes)],
                       [min(counts), max(counts)])
    ]
    plt.legend(handles=legend_elements,
              loc='upper right',
              title='Gene Count')
    
    plt.title('GO Terms Interaction Network\n(Emmaplot Style)',
             pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    return plt

def create_go_graph_rpy2(df, aspect='BP', max_nodes=50, save_path=None):
    """
    Utiliza rpy2 para generar un gráfico GOgraph con datos de Python.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con resultados de análisis GO.
        Debe contener columnas: 'ID', 'Description', 'p.adjust', 'Count'
    aspect : str
        GO aspect ('BP', 'MF', 'CC').
    max_nodes : int
        Número máximo de nodos a mostrar en el gráfico.
    save_path : str, optional
        Ruta para guardar el gráfico. Si es None, se muestra en pantalla.
    """
    try:
        # Configurar entorno R
        #base, utils, go_db, gostats, graph, rgraphviz = setup_r_environment()
        
        # Convertir el DataFrame a formato R
        go_ids = StrVector(df['ID'].tolist())
        
        # Crear el código R para la visualización
        r_code = """
        function(go_ids, aspect, max_nodes, save_path) {
            library(GO.db)
            library(GOstats)
            library(graph)
            library(Rgraphviz)
            
            # Crear el grafo GO
            go_terms <- unique(go_ids)
            gograph <- GOGraph(go_terms, GOBPPARENTS)
            
            # Limitar el número de nodos
            if (numNodes(gograph) > max_nodes) {
                gograph <- subGraph(sample(nodes(gograph), max_nodes), gograph)
            }
            
            # Generar el gráfico
            if (!is.null(save_path)) {
                pdf(save_path)
            }
            
            plot(gograph, main = paste("GO", aspect, "Graph"))
            
            if (!is.null(save_path)) {
                dev.off()
            }
        }
        """
        
        # Crear la función R y ejecutarla
        r_func = robjects.r(r_code)
        r_func(go_ids, aspect, max_nodes, save_path)
        
        print(f"Gráfico GO creado exitosamente{' y guardado en ' + save_path if save_path else ''}")
        
    except Exception as e:
        raise Exception(f"Error creando el gráfico GO: {str(e)}")
