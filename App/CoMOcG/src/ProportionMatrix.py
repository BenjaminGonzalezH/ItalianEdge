######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from scipy.sparse import csr_matrix                 # Compresed space row matrix.
from typing import Tuple                            # Multiple returns doc.
import os                                           # OS callings.
import matplotlib.pyplot as plt                     # Graph construction.

######### Functions #########
def ProportionsMatrix(summed_matrix: csr_matrix
                      ) -> Tuple[list[np.ndarray], list[np.ndarray]] :
    """
    ProportionMatrix_Similarity(function): Proportion Matrix tell us % of cases where a pair
    of index (genes) have the same number (are in the same cluster). Also, % where they are not.
        
    Parameters:
    - summed_matrix (csr_matrix): Sum of a list of sparce matrix in csr format.
    - total_solutionns (int): Amount of solutions to study.

    Returns:
    - Proportion_matrix (list[ndarray]): Matrix that represent the proportion of solutions where a pair 
    of genes are allocates in the same cluster.
    - Distance_matrix (list[ndarray]): Matrix that represent the proportion of solutions where a pair 
    of genes are allocates in the same cluster.
    """
    # Check if summed_matrix is in CSR format
    if not isinstance(summed_matrix, csr_matrix):
        raise TypeError("The input matrix has to be in csr format.")
    
    try:
        with np.errstate(divide='ignore', invalid='ignore'):  # ignote invalid divitions.
            Proportion_matrix = summed_matrix.toarray() / summed_matrix[0, 0]
            Proportion_matrix[np.isnan(Proportion_matrix)] = 0  # Replace NaN with 0's
            Proportion_matrix[np.isinf(Proportion_matrix)] = 0  # Replace NaN with 0's

        Distance_matrix = 1 - Proportion_matrix
        return Proportion_matrix, Distance_matrix

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None

def save_matrices(proportion_matrix: np.ndarray, 
                  distance_matrix: np.ndarray, 
                  proportion_filepath: str, 
                  distance_filepath: str) -> None:
    """
    save_matrices (function): Save the Proportion and Distance matrices to files.

    Parameters:
    - proportion_matrix (np.ndarray): Matrix representing proportions of similarity.
    - distance_matrix (np.ndarray): Matrix representing proportions of dissimilarity.
    - proportion_filepath (str): Path to save the proportion matrix.
    - distance_filepath (str): Path to save the distance matrix.
    """
    try:
        # Create directories if needed
        for filepath in [proportion_filepath, distance_filepath]:
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
        
        # Save proportion matrix
        np.savetxt(proportion_filepath, proportion_matrix, delimiter=",", fmt="%.6f")
        print(f"Proportion matrix saved at: {proportion_filepath}")
        
        # Save distance matrix
        np.savetxt(distance_filepath, distance_matrix, delimiter=",", fmt="%.6f")
        print(f"Distance matrix saved at: {distance_filepath}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")

def plot_and_save_heatmaps(proportion_matrix: np.ndarray, 
                           distance_matrix: np.ndarray, 
                           proportion_title: str = "Proportion Matrix Heatmap", 
                           distance_title: str = "Distance Matrix Heatmap", 
                           save_path: str = "heatmaps", 
                           resolution: int = 300, 
                           figsize: tuple = (16, 8)) -> None:
    """
    Plot heatmaps for the Proportion and Distance matrices and save them as PNG files.
    
    Parameters:
    - proportion_matrix (np.ndarray): Matrix representing proportions of similarity.
    - distance_matrix (np.ndarray): Matrix representing proportions of dissimilarity.
    - proportion_title (str): Title for the proportion matrix heatmap. Default is "Proportion Matrix Heatmap".
    - distance_title (str): Title for the distance matrix heatmap. Default is "Distance Matrix Heatmap".
    - save_path (str): Directory to save the PNG files.
    - resolution (int): Resolution of the saved images in DPI. Default is 300.
    - figsize (tuple): Figure size for the heatmaps. Default is (16, 8).
    """
    try:
        # Create the directory if it doesn't exist
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Set up the figure
        plt.figure(figsize=figsize)
        
        # Heatmap for Proportion Matrix
        plt.subplot(1, 2, 1)
        plt.imshow(proportion_matrix, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Proportion Value")
        plt.title(proportion_title)
        plt.xlabel("Genes")
        plt.ylabel("Genes")
        
        # Heatmap for Distance Matrix
        plt.subplot(1, 2, 2)
        plt.imshow(distance_matrix, cmap="plasma", interpolation="nearest")
        plt.colorbar(label="Distance Value")
        plt.title(distance_title)
        plt.xlabel("Genes")
        plt.ylabel("Genes")
        
        # Save the figure as a PNG file
        png_path = os.path.join(save_path, "heatmaps.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=resolution)
        print(f"Heatmaps saved successfully at: {png_path}")
        
        # Show the plot
        plt.show()
    
    except Exception as e:
        print(f"Unexpected error while plotting and saving heatmaps: {e}")