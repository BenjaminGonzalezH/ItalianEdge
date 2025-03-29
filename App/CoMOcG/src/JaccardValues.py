######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import rand_score
import itertools
import pandas as pd

######### Functions #########

"""
This block contains all main functions.
"""

def compute_jaccard(A, B):
    A = np.array(A)
    B = np.array(B)
    n = len(A)

    # Creamos matrices booleanas de comparación
    same_A = (A[:, None] == A[None, :])
    same_B = (B[:, None] == B[None, :])

    # Tomamos solo la parte superior triangular para evitar duplicados (i<j)
    upper_triangle = np.triu_indices(n, k=1)

    same_A_upper = same_A[upper_triangle]
    same_B_upper = same_B[upper_triangle]

    # Condiciones para r, u, v
    r = np.sum(same_A_upper & same_B_upper)
    u = np.sum(same_A_upper & ~same_B_upper)
    v = np.sum(~same_A_upper & same_B_upper)

    if (r + u + v) == 0:
        return 0

    return r / (r + u + v)

def process_JaccardValues(
        Matrix: list[np.ndarray], 
        n_threads: int) -> np.ndarray:
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

def Jaccar_similarityClusters(Solution1: list[set], Solution2: list[set]) -> np.ndarray:
    """
    Compute the Jaccard similarity matrix for two clustering solutions.

    Parameters:
    - Solution1 (list[set]): Clusters of the first solution as sets.
    - Solution2 (list[set]): Clusters of the second solution as sets.

    Returns:
    - MatrixJaccard (np.ndarray): Jaccard similarity matrix for the solutions' clusters.
    """
    # Construction of jaccard matrix.
    n1 = len(Solution1)  # Number of clusters in Solution1
    n2 = len(Solution2)  # Number of clusters in Solution2
    MatrixJaccard = np.zeros((n1, n2))  # Create a matrix with dimensions n1 x n2

    # Compare clusters of Solution1 with Solution2
    for i in range(n1):
        for j in range(n2):
            union = len(Solution1[i] | Solution2[j])  # Union of the two sets
            intersection = len(Solution1[i] & Solution2[j])  # Intersection of the two sets
            if union == 0:
                MatrixJaccard[i, j] = 0  # If there is no union, similarity is 0
            else:
                Jaccard = intersection / union  # Jaccard similarity
                MatrixJaccard[i, j] = Jaccard

    return MatrixJaccard

def compute_ari(Solution1: np.ndarray, Solution2: np.ndarray) -> float:
    """
    compute_ari(function): Compute the Adjusted Rand Index (ARI) for two solutions.

    Parameters:
    - Solution1 (ndarray): First solution vector (cluster assignments).
    - Solution2 (ndarray): Second solution vector (cluster assignments).

    Returns:
    - ari_index (float): Adjusted Rand Index between the two vectors.
    """
    # Check length.
    if len(Solution1) != len(Solution2):
        raise ValueError("Both vectors must have the same length.")
    
    return adjusted_rand_score(Solution1, Solution2)

def process_ARIValues(Matrix: list[np.ndarray], n_threads: int) -> np.ndarray:
    """
    process_ARIValues(function): Compute the pairwise Adjusted Rand Index matrix for a set of solutions.

    Parameters:
    - Matrix (list[np.ndarray]] or np.ndarray): Matrix of cluster solutions.
    - n_threads (int): Number of threads to use for parallel computation.

    Returns:
    - ARI_Matrix (np.ndarray): Pairwise Adjusted Rand Index matrix for the solutions.
    """
    # Check input.
    if isinstance(Matrix, list):
        if not all(isinstance(row, np.ndarray) for row in Matrix):
            raise ValueError("Matrix has to be a list of ndarrays.")
    elif isinstance(Matrix, np.ndarray):
        if Matrix.ndim != 2:
            raise ValueError("Matrix must be a 2D NumPy array")
    else:
        raise ValueError("Matrix has to be a list of ndarrays or a 2D NumPy array.")
    
    if n_threads <= 0:
        raise ValueError("n_threads must be a positive integer.")
    
    # Create an Adjusted Rand Index Matrix.
    n_solutions = len(Matrix)
    ARI_Matrix = np.zeros((n_solutions, n_solutions))

    # Aux function to handle solution's index.
    def compute_pairwise_ari(indices):
        i, j = indices
        return i, j, compute_ari(Matrix[i], Matrix[j])
    
    # Create Combination of indices.
    indices = [(i, j) for i in range(n_solutions) for j in range(i, n_solutions)]
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(compute_pairwise_ari, indices))
    
    # Fill symmetric ARI values matrix.
    for i, j, ari_value in results:
        ARI_Matrix[i, j] = ari_value
        ARI_Matrix[j, i] = ari_value
    
    return ARI_Matrix

def compute_ri(Solution1: np.ndarray, Solution2: np.ndarray) -> float:
    """
    compute_ri(function): Compute the Rand Index (RI) for two solutions.

    Parameters:
    - Solution1 (ndarray): First solution vector (cluster assignments).
    - Solution2 (ndarray): Second solution vector (cluster assignments).

    Returns:
    - ri_index (float): Rand Index between the two vectors.
    """
    # Check length.
    if len(Solution1) != len(Solution2):
        raise ValueError("Both vectors must have the same length.")
    
    return rand_score(Solution1, Solution2)

def process_RIValues(Matrix: list[np.ndarray], n_threads: int) -> np.ndarray:
    """
    process_RIValues(function): Compute the pairwise Rand Index matrix for a set of solutions.

    Parameters:
    - Matrix (list[np.ndarray]] or np.ndarray): Matrix of cluster solutions.
    - n_threads (int): Number of threads to use for parallel computation.

    Returns:
    - RI_Matrix (np.ndarray): Pairwise Rand Index matrix for the solutions.
    """
    # Check input.
    if isinstance(Matrix, list):
        if not all(isinstance(row, np.ndarray) for row in Matrix):
            raise ValueError("Matrix has to be a list of ndarrays.")
    elif isinstance(Matrix, np.ndarray):
        if Matrix.ndim != 2:
            raise ValueError("Matrix must be a 2D NumPy array")
    else:
        raise ValueError("Matrix has to be a list of ndarrays or a 2D NumPy array.")
    
    if n_threads <= 0:
        raise ValueError("n_threads must be a positive integer.")
    
    # Create Rand Index Matrix.
    n_solutions = len(Matrix)
    RI_Matrix = np.zeros((n_solutions, n_solutions))

    # Aux function to handle solution's index.
    def compute_pairwise_ri(indices):
        i, j = indices
        return i, j, compute_ri(Matrix[i], Matrix[j])
    
    # Create Combination of indices.
    indices = [(i, j) for i in range(n_solutions) for j in range(i, n_solutions)]
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(compute_pairwise_ri, indices))
    
    # Fill symmetric RI matrix.
    for i, j, ri_value in results:
        RI_Matrix[i, j] = ri_value
        RI_Matrix[j, i] = ri_value
    
    return RI_Matrix

def compare_solution_pair(idx1, idx2, solutions):
    """
    Compares two solutions and returns the equivalence pairs.
    
    Parameters:
    - idx1 (int): Index of the first solution.
    - idx2 (int): Index of the second solution.
    - solutions (list): List of solutions (list of sets).
    
    Returns:
    - equivalent_pairs (list): List of tuples (cluster indices, similarity).
    """
    # Obtener la matriz de Jaccard entre las dos soluciones
    MatrixJaccard = Jaccar_similarityClusters(solutions[idx1], solutions[idx2])

    # Crear una lista de tuplas con los valores y sus índices
    similarity_pairs = [
        (i, j, MatrixJaccard[i, j])
        for i in range(len(solutions[idx1]))
        for j in range(len(solutions[idx2]))
    ]
    
    # Ordenar las similitudes de mayor a menor
    similarity_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Seleccionar los grupos equivalentes, asegurando que no se repitan
    equivalent_pairs = []
    used_clusters_solution1 = set()
    used_clusters_solution2 = set()

    for i, j, similarity in similarity_pairs:
        if i not in used_clusters_solution1 and j not in used_clusters_solution2:
            equivalent_pairs.append((i, j, similarity))
            used_clusters_solution1.add(i)
            used_clusters_solution2.add(j)

    return equivalent_pairs

def find_equivalent_clusters(solutions: list[list[set]]) -> pd.DataFrame:
    """
    Encuentra los grupos equivalentes entre varias soluciones de agrupamiento usando las matrices de similitudes de Jaccard.

    Parameters:
    - solutions (list[list[set]]): Lista de soluciones, cada una representada por una lista de conjuntos de clústeres.

    Returns:
    - pd.DataFrame: Un DataFrame con las combinaciones de grupos equivalentes entre las soluciones.
    """
    all_equivalent_pairs = []

    # Usar ThreadPoolExecutor para paralelizar el cálculo
    with ThreadPoolExecutor() as executor:
        # Generar todas las combinaciones de soluciones para comparar
        future_to_comparison = {
            executor.submit(compare_solution_pair, idx1, idx2, solutions): (idx1, idx2)
            for idx1 in range(len(solutions))
            for idx2 in range(idx1 + 1, len(solutions))
        }
        
        # Obtener los resultados y agregarlos
        for future in future_to_comparison:
            equivalent_pairs = future.result()
            idx1, idx2 = future_to_comparison[future]
            # Almacenar el par de soluciones, los clusters equivalentes y las similitudes Jaccard
            all_equivalent_pairs.append(((idx1, idx2), 
                                         [(pair[0], pair[1]) for pair in equivalent_pairs],
                                         [pair[2] for pair in equivalent_pairs]))

    # Crear el DataFrame de resultados
    df = pd.DataFrame(all_equivalent_pairs, columns=["Solution Pair", "Equivalent Clusters", "Jaccard Similarities"])

    return df