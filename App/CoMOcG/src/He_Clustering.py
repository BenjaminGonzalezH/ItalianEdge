######### Libraries #########
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import os
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist

######### Functions #########

def He_clustering(ProportionMatrix, genes, num_groups=4, dendrogram_file="dendrogram.png", show_flag=1):
    """
    He_clustering(function)
        Input:
            - ProportionMatrix: Result from 'ProportionMatrix_Similarity'
            function that creates a new solution.
            - num_groups: number of cluster for the consensus solution.
            - dendrogram_file: Name of the solution.
            - show_flag: flag that indicates the desire of ploting the
            debndogram result.
        Output:
            - ConsensusSolution: Consensus cluster for comparitions.
        
        Description: Creation of the consensus cluster thar represents the
        performances of all solutions.
    """
    try:
        # Convert matrix into a condensed matrix (list
        # of pairs with distance of each pair of
        # elements).
        condensed_dist_matrix = squareform(ProportionMatrix)

        # Create the heiracial cluster.
        Z = linkage(condensed_dist_matrix, method='single')

        # Define the consensus cluster.
        ConsensusSolution = fcluster(Z, num_groups, criterion='maxclust')

        # Plot Dendogram.
        plt.figure(figsize=(80, 35))
        dendro = dendrogram(Z, labels=genes)

        # Add cut line.
        plt.axhline(y=Z[-(num_groups - 1), 2], c='red', linestyle='--')
        plt.title(f'Dendrograma con {num_groups} grupos')
        plt.xlabel('Genes')
        plt.ylabel('Distance')
        if(show_flag == 1):
            plt.show()

        # Obtain Downloads dir of the user computer and put
        # the final image on that.
        downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
        plt.savefig(downloads_dir + "\\" + dendrogram_file)
        plt.close()
        
        return list(ConsensusSolution)

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None
    
def consensus_cluster_function(consensus_matrix_result, num_cluster):
    # Convertir la matriz de consenso a formato denso y asegurar la diagonal a 0
    consensus_matrix_dense = consensus_matrix_result.toarray()
    np.fill_diagonal(consensus_matrix_dense, 1)
    
    # Crear el gráfico de calor de la matriz de consenso
    plt.figure(figsize=(10, 10))
    sns.heatmap(consensus_matrix_dense, cmap="viridis", square=True, cbar=True)
    plt.title("Matriz de Consenso")
    plt.savefig("levelplot.pdf", bbox_inches='tight')
    plt.close()
    
    # Convertir la matriz de consenso en una matriz de distancia
    dist_matrix = 1 - consensus_matrix_dense
    dist_array = squareform(dist_matrix)  # Convertir a formato condensado para `linkage`
    
    # Realizar el clustering jerárquico usando el método 'single'
    hclust_avg = linkage(dist_array, method="single")
    
    # Generar el dendrograma y guardarlo como PDF
    plt.figure(figsize=(10, 10))
    dendrogram(hclust_avg, color_threshold=0.7 * max(hclust_avg[:, 2]), orientation="top")
    plt.title("Dendrograma de la Matriz de Consenso")
    plt.savefig("dendrograma.pdf", bbox_inches='tight')
    plt.close()
    
    # Realizar el corte de árbol para definir los clusters
    consensus_cluster = fcluster(hclust_avg, num_cluster, criterion='maxclust')
    
    # Guardar los resultados en un archivo CSV
    pd.DataFrame(consensus_cluster, columns=["Cluster"]).to_csv("clusterConsenso.csv", index=False)
    
    return consensus_cluster