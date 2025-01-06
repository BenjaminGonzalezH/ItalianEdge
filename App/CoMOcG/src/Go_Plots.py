import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx # type: ignore
from matplotlib.patches import Circle
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from GoEnrischment import setup_r_environment
import seaborn as sns

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
        plt.figure(figsize=(12, 8))
        sorted_df = wang_similarity_df.sort_values("GeneRatio", ascending=False)
        size = sorted_df["Count"]  # Assuming 'Count' column represents the number of genes
        color = sorted_df["p.adjust"]  # Assuming 'p.adjust' column represents the adjusted p-values
        
        scatter = plt.scatter(
            x=sorted_df["GeneRatio"],
            y=sorted_df["Description"],
            s=size * 10,  # Scale size for better visualization
            c=color,
            cmap="coolwarm",  # Color map ranging from blue to red
            alpha=0.7,
            edgecolors="w",
            linewidth=0.5
        )
        
        plt.xlabel("Gene Ratio")
        plt.ylabel("GO Terms")
        plt.title("Gene Ratio for GO Terms")
        plt.colorbar(scatter, label="Adjusted p-value")
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
            G.add_node(go_term, label=row['Description'], size=row['Count'])
            
            for _, other_row in wang_similarity_df.iterrows():
                if row['ID'] != other_row['ID'] and row['wang_similarity'] >= threshold:
                    G.add_edge(row['ID'], other_row['ID'], weight=row['wang_similarity'])
        
        # Draw the network
        pos = nx.spring_layout(G)
        sizes = [G.nodes[node]['size'] * 100 for node in G.nodes]  # Scale size for better visualization
        widths = [G[u][v]['weight'] * 10 for u, v in G.edges]  # Scale width for better visualization
        
        plt.figure(figsize=(14, 14))
        nx.draw(
            G, pos, with_labels=True, node_size=sizes, node_color="skyblue",
            font_size=10, font_color="black", edge_color="gray", width=widths
        )
        plt.title("GO Interaction Network")
        plt.show()
    except Exception as e:
        print(f"Error plotting GO interaction network: {e}")

def plot_go_interaction_network_rpy2(df, organism="org.Hs.eg.db", aspect="BP", similarity_threshold=0.7, save_path=None):
    """
    plot_go_interaction_network_rpy2(function): 
    Generate an interaction network for GO terms using R and rpy2.

    Parameters:
    - df (pd.DataFrame): DataFrame with GO terms, their descriptions, and Wang similarity scores.
    - organism (str): Organism database to use in R.
    - aspect (str): GO aspect to focus on ('BP', 'MF', 'CC').
    - similarity_threshold (float): Minimum Wang similarity to create an edge.
    - save_path (str): Path to save the graph. If None, the graph is displayed in R.

    Returns:
    - None
    """
    try:
        # Convert GO terms to R vector
        go_ids = robjects.StrVector(df['ID'].tolist())
        descriptions = robjects.StrVector(df['Description'].tolist())

        # Create R code for network generation
        r_code = f"""
        function(go_ids, descriptions, aspect, similarity_threshold, save_path) {{
            library({organism})
            library(GOSemSim)
            library(igraph)
            
            # Prepare GO data
            go_db <- godata(annoDb = "{organism}", ont = aspect, computeIC = TRUE)
            sim_matrix <- mgoSim(GO1 = go_ids, GO2 = go_ids, semData = go_db, measure = "Wang", combine = NULL)
            
            # Filter edges based on similarity threshold
            edge_list <- which(sim_matrix >= similarity_threshold, arr.ind = TRUE)
            edge_list <- edge_list[edge_list[,1] != edge_list[,2], ]  # Remove self-loops
            
            # Create igraph object
            vertex_df <- data.frame(name = go_ids, description = descriptions)
            edge_list_df <- as.data.frame(edge_list)
            edge_list_df <- edge_list_df[edge_list_df[,1] %in% vertex_df$name & edge_list_df[,2] %in% vertex_df$name, ]
            g <- graph_from_data_frame(edge_list_df, directed = FALSE, vertices = vertex_df)
            
            # Plot the network
            if (!is.null(save_path)) {{
                pdf(save_path)
            }}
            plot(g, vertex.label = V(g)$description, vertex.label.cex = 0.8, vertex.label.color = "black", vertex.size = 10, 
                 edge.width = E(g)$weight * 5, main = paste("GO Interaction Network (", aspect, ")"))
            
            if (!is.null(save_path)) {{
                dev.off()
            }}
        }}
        """

        # Execute the R function
        r_func = robjects.r(r_code)
        r_func(go_ids, descriptions, aspect, similarity_threshold, save_path)

        print(f"GO interaction network created successfully{' and saved at ' + save_path if save_path else ''}")
    
    except Exception as e:
        print(f"Error generating GO interaction network: {str(e)}")

def plot_go_tree_rpy2(df, organism="org.Hs.eg.db", aspect="BP", max_nodes=50, save_path=None):
    """
    plot_go_tree_rpy2(function): 
    Generate a GO tree visualization using R and rpy2 with enhanced visual elements.

    Parameters:
    - df (pd.DataFrame): DataFrame containing GO terms with 'ID' and 'Description' columns.
    - organism (str): Organism database to use in R.
    - aspect (str): GO aspect to focus on ('BP', 'MF', 'CC').
    - max_nodes (int): Maximum number of nodes to display in the tree.
    - save_path (str): Path to save the tree. If None, the tree is displayed in R.

    Returns:
    - None
    """
    try:
        # Convert GO terms to R vector
        go_ids = robjects.StrVector(df['ID'].tolist())

        # Create R code for tree generation
        r_code = f"""
        function(go_ids, aspect, max_nodes, save_path) {{
            library(GO.db)
            library(GOstats)
            library(graph)
            library(Rgraphviz)
            
            # Create the GO graph
            gograph <- GOGraph(go_ids, GOBPPARENTS)
            
            # Reduce the graph size if necessary
            if (numNodes(gograph) > max_nodes) {{
                gograph <- subGraph(sample(nodes(gograph), max_nodes), gograph)
            }}
            
            # Customize node attributes
            attrs <- list()
            attrs$node <- list(fillcolor = "lightblue", shape = "ellipse", fontsize = 10)
            attrs$edge <- list(color = "gray", arrowsize = 0.5)
            
            # Plot the graph
            if (!is.null(save_path)) {{
                pdf(save_path)
            }}
            
            plot(gograph, attrs = attrs, main = paste("GO Term Tree (", aspect, ")"))
            
            if (!is.null(save_path)) {{
                dev.off()
            }}
        }}
        """

        # Execute the R function
        r_func = robjects.r(r_code)
        r_func(go_ids, aspect, max_nodes, save_path)

        print(f"GO tree created successfully{' and saved at ' + save_path if save_path else ''}")
    
    except Exception as e:
        print(f"Error generating GO tree: {str(e)}")