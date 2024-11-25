######### Libraries #########
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import os
import pandas as pd
import seaborn as sns
import numpy as np

######### Functions #########

def He_clustering(ProportionMatrix, genes, num_groups=4, dendrogram_file="dendrogram.png", show_flag=1):
    """
    He_clustering(function)
        Input:
            - ProportionMatrix: Result from 'ProportionMatrix_Similarity'
            function that creates a new solution.
            - genes: Genes identificators.
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
    