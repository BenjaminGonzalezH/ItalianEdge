######### Libraries #########
import numpy as np                                                  # Efficient Math Operations.
import matplotlib.pyplot as plt                                     # Graph construction.
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster   # Create clustering.
from scipy.spatial.distance import squareform                       # Create dendogram.
import os                                                           # OS callings.

######### Functions #########

"""
This block contains all main functions.
"""

def He_clustering(
        distance_matrix: np.ndarray, 
        genes: list[str], 
        num_groups: int = 4,
        save_path: str = "dendrogram", 
        dendrogram_file: str = "dendrogram.png", 
        show_flag: bool = True) -> list:
    """
    He_clustering(function): Perform hierarchical clustering and generate a dendrogram with labels
    showing gene names and their positions in the list.

    Parameters:
    - distance_matrix (np.ndarray): Square distance matrix between genes.
    - genes (list[str]): Gene identifiers.
    - num_groups (int): Number of clusters for the consensus solution.
    - save_path (str): Directory to save the dendrogram.
    - dendrogram_file (str): Name of the output file.
    - show_flag (bool): Whether to display the dendrogram. Default is True.
        
    Returns:
    - list: Consensus clustering solution.
    """
    try:
        # Convert the matrix to condensed form if necessary
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("Distance matrix must be square.")
        
        condensed_dist_matrix = squareform(distance_matrix)

        # Perform hierarchical clustering
        Z = linkage(condensed_dist_matrix, method='single')

        # Define the consensus clusters
        consensus_solution = fcluster(Z, num_groups, criterion='maxclust')

        # Generate labels with gene names and their positions
        labels = [f"{i}-{gene}" for i, gene in enumerate(genes)]

        # Plot the dendrogram
        plt.figure(figsize=(20, 10))
        dendro = dendrogram(Z, labels=labels)

        # Add a horizontal line for cluster separation
        plt.axhline(y=Z[-(num_groups - 1), 2], c='red', linestyle='--')
        plt.title(f'Dendrogram with {num_groups} clusters')
        plt.xlabel('Genes (Index-Name)')
        plt.ylabel('Distance')

        # Save the dendrogram
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, dendrogram_file), dpi=300)
        print(f"Dendrogram saved at: {os.path.join(save_path, dendrogram_file)}")
        
        # Show the plot if flag is True
        if show_flag:
            plt.show()
        plt.close()

        return list(consensus_solution)

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None