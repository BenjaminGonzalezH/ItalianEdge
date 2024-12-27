import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
        # Create a matrix of genes x genes based on GO similarity
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
                                (wang_similarity_df['GO1'] == go_i) &
                                (wang_similarity_df['GO2'] == go_j),
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