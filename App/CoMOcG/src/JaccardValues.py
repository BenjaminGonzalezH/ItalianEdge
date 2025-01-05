######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.
import matplotlib.pyplot as plt                     # Graph construction.
import pandas as pd                                 # Handle dataframe.

######### Functions #########

def compute_jaccard(Solution1: np.ndarray, Solution2: np.ndarray) -> float:
    """
    compute_jaccard(function): Compute the Jaccard similarity index for two solutions.

    Parameters:
    - Solution1 (ndarray): First solution vector.
    - Solution2 (ndarray): Second solution vector.

    Returns:
    - jaccard_index (float): Jaccard similarity index between the two vectors.
    """
    # Check length.
    if len(Solution1) != len(Solution2):
        raise ValueError("Ambos vectores deben tener la misma longitud.")
    
    # Jaccard Value calculus.
    intersection = np.sum(np.minimum(Solution1, Solution2))
    union = np.sum(np.maximum(Solution1, Solution2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def process_JaccardValues(Matrix: list[np.ndarray], n_threads: int):
    """
    process_JaccardValues(function): Compute the pairwise Jaccard similarity matrix for a set of solutions.

    Parameters:
    - Matrix (list[np.ndarray]] or np.ndarray): Matrix of cluster solutions.
    - n_threads (int): Number of threads to use for parallel computation.

    Returns:
    - Jaccard_Matrix (np.ndarray): Pairwise Jaccard similarity matrix for the solutions.
    """
    # Check input.
    if isinstance(Matrix, list):
        if not all(isinstance(row, list) for row in Matrix):
            raise ValueError("Matrix has to be a list of ndarrays.")
    elif isinstance(Matrix, np.ndarray):
        if Matrix.ndim != 2:
            raise ValueError("Matrix is Numpy array 2D")
    else:
        raise ValueError("Matrix has to be a list of ndarrays or NumPy array 2D.")
    
    if n_threads <= 0:
        raise ValueError("n_threads must be a positive integer.")
    
    # Create a Jaccard Matrix.
    n_solutions = len(Matrix)
    Jaccard_Matrix = np.zeros((n_solutions, n_solutions))

    # Aux function to handle solution's index.
    def compute_pairwise_proportion(indices):
        i, j = indices
        return i, j, compute_jaccard(Matrix[i], Matrix[j])
    
    # Create Combination of index.
    indices = [(i, j) for i in range(n_solutions) for j in range(i, n_solutions)]
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(compute_pairwise_proportion, indices))
    
    # Fill symetric jaccard values matrix.
    for i, j, proportion in results:
        Jaccard_Matrix[i, j] = proportion
        Jaccard_Matrix[j, i] = proportion
    
    return Jaccard_Matrix

def Jaccar_similarityClusters(Solution1: list[set], Solution2: list[set]):
    """
    Jaccar_similarityClusters(function): Compute the Jaccard similarity matrix for
    two specifict solutions, this works to recognize similar clusters between two
    solutiions.

    Parameters:
    - Solution1 (list[set]): Clusters of the first solution as sets.
    - Solution2 (list[set]): Clusters of the second solution as sets.

    Returns:
    - MatrixJaccard (np.ndarray): Jaccard similarity matrix for the 
    solution's clusters.
    """
    # Construction of jaccard matrix.
    n = len(Solution1)
    MatrixJaccard = np.zeros((n, n))

    # Compare clusters of two solutions.
    for i in range(len(Solution1)):
        for j in range(i, len(Solution2)):
            union = len(Solution1[i] | Solution2[j])
            intersection = len(Solution1[i] & Solution2[j])
            if union == 0:
                MatrixJaccard[i, j] = 0
                MatrixJaccard[j, i] = 0
            else:
                Jaccard = intersection / union
                MatrixJaccard[i, j] = Jaccard
                MatrixJaccard[j, i] = Jaccard

    return MatrixJaccard

def plot_jaccard_heatmap(jaccard_matrix: np.ndarray, 
                         title: str = "Jaccard Similarity Heatmap", 
                         save_path: str = None, 
                         resolution: int = 300, 
                         figsize: tuple = (10, 8)) -> None:
    """
    plot_jaccard_heatmap(function): Plot and optionally save a heatmap 
    for the Jaccard similarity matrix.

    Parameters:
    - jaccard_matrix (np.ndarray): Jaccard similarity matrix.
    - title (str): Title of the heatmap. Default is "Jaccard Similarity Heatmap".
    - save_path (str): Path to save the heatmap as a PNG file. If None, the heatmap is not saved.
    - resolution (int): Resolution of the saved PNG file in DPI. Default is 300.
    - figsize (tuple): Figure size for the heatmap. Default is (10, 8).
    
    Returns:
    - None. Displays and optionally saves the heatmap.
    """
    try:
        # Create figure.
        plt.figure(figsize=figsize)
        heatmap = plt.imshow(jaccard_matrix, cmap="cividis", interpolation="nearest")
        plt.colorbar(heatmap, label="Jaccard Similarity")

        # Add integer ticks for X and Y axes.
        num_solutions = jaccard_matrix.shape[0]  # Assuming square matrix
        plt.xticks(ticks=np.arange(num_solutions), labels=np.arange(1, num_solutions + 1))
        plt.yticks(ticks=np.arange(num_solutions), labels=np.arange(1, num_solutions + 1))
        
        # Add labels.
        plt.title(title)
        plt.xlabel("Solutions")
        plt.ylabel("Solutions")
        
        # Save figure in filepath.
        if save_path:
            plt.savefig(save_path, dpi=resolution, bbox_inches="tight")
            print(f"Heatmap guardado en: {save_path}")
        
        # Show heatmap.
        plt.show()
        plt.close()
    
    except Exception as e:
        print(f"Ocurrió un error al graficar el heatmap: {e}")

def save_jaccard_matrix(jaccard_matrix: np.ndarray, filepath: str) -> None:
    """
    save_jaccard_matrix(function): Save the Jaccard similarity matrix to a CSV file.

    Parameters:
    - jaccard_matrix (np.ndarray): Jaccard similarity matrix.
    - filepath (str): Path to save the CSV file.
    
    Returns:
    - None. Saves the matrix to a file.
    """
    try:
        # Convertir la matriz en un DataFrame
        df = pd.DataFrame(jaccard_matrix)
        
        # Guardar el DataFrame como CSV
        df.to_csv(filepath, index=False, header=False)
        print(f"Matriz de similitud Jaccard guardada en: {filepath}")
    
    except Exception as e:
        print(f"Ocurrió un error al guardar la matriz Jaccard: {e}")
