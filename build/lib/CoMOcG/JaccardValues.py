######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.
from sklearn.metrics import adjusted_rand_score     # Pre-build ARI function.
from sklearn.metrics import rand_score              # Pre-build RI function.
import pandas as pd                                 # Dataframe managment.

######### Functions #########

"""
This block contains all main functions.
"""

def compute_jaccard(
        A: np.ndarray, 
        B: np.ndarray
        ) -> float:
    """
    compute_jaccard (function): Compute the Jaccard index between two cluster label vectors.
    
    Parameters:
    - A: Iterable of cluster labels (e.g., list, np.ndarray).
    - B: Iterable of cluster labels (same length as A).
    
    Returns:
    - jaccard index: Jaccard similarity index.
    """
    try:
        # Convert inputs to NumPy arrays
        A = np.ravel(np.array(A))
        B = np.ravel(np.array(B))

        # Validate shape
        if A.shape != B.shape:
            raise ValueError("Input arrays A and B must have the same shape.")

        n = len(A)
        if n < 2:
            raise ValueError("Input arrays must contain at least two elements to compute pairwise comparisons.")

        # Build pairwise comparison matrices.
        same_A = (A[:, None] == A[None, :])
        same_B = (B[:, None] == B[None, :])

        # Use upper triangle indices (excluding diagonal).
        upper_triangle = np.triu_indices(n, k=1)
        same_A_upper = same_A[upper_triangle]
        same_B_upper = same_B[upper_triangle]

        # Compute intersection and differences.
        r = np.sum(same_A_upper & same_B_upper)
        u = np.sum(same_A_upper & ~same_B_upper)
        v = np.sum(~same_A_upper & same_B_upper)

        # Avoid division by zero.
        denominator = r + u + v
        if denominator == 0:
            return 0.0

        return r / denominator

    except Exception as e:
        print(f"Error in compute_jaccard: {e}")
        return None

def process_JaccardValues(
        Matrix: list[np.ndarray], 
        n_threads: int
        ) -> np.ndarray:
    """
    process_JaccardValues(function): Compute the pairwise Jaccard similarity matrix for a set of solutions.

    Parameters:
    - Matrix (list[np.ndarray]] or np.ndarray): Matrix of cluster solutions.
    - n_threads (int): Number of threads to use for parallel computation.

    Returns:
    - Jaccard_Matrix (np.ndarray): Pairwise Jaccard similarity matrix for the solutions.
    """
    try:
        # Validation of thread number.
        if not isinstance(n_threads, int) or n_threads <= 0:
            raise ValueError("n_threads must be a positive integer.")

        # Check input matrix.
        if isinstance(Matrix, list):
            if not all(isinstance(row, (np.ndarray, list)) for row in Matrix):
                raise ValueError("All elements in the list must be NumPy arrays or lists.")
            Matrix = [np.array(row) for row in Matrix]
        elif isinstance(Matrix, np.ndarray):
            if Matrix.ndim != 2:
                raise ValueError("Matrix must be a 2D NumPy array.")
            # Convert each row to a separate vector.
            Matrix = [Matrix[i, :] for i in range(Matrix.shape[0])]
        else:
            raise TypeError("Matrix must be a list of NumPy arrays or a 2D NumPy array.")

        n_solutions = len(Matrix)
        if n_solutions == 0:
            raise ValueError("Matrix must contain at least one solution.")

        # Crear la matriz de Jaccard
        Jaccard_Matrix = np.zeros((n_solutions, n_solutions))

        # Función auxiliar
        def compute_pairwise_proportion(indices):
            i, j = indices
            return i, j, compute_jaccard(Matrix[i], Matrix[j])

        # Índices del triángulo superior (incluyendo diagonal)
        indices = [(i, j) for i in range(n_solutions) for j in range(i, n_solutions)]

        # Cálculo paralelo
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            results = list(executor.map(compute_pairwise_proportion, indices))

        # Llenar la matriz simétrica
        for i, j, proportion in results:
            Jaccard_Matrix[i, j] = proportion
            Jaccard_Matrix[j, i] = proportion

        return Jaccard_Matrix

    except Exception as e:
        print(f"Error in process_JaccardValues: {e}")
        return np.array([])

def Jaccar_similarityClusters(
        Solution1: list[set], 
        Solution2: list[set]
        ) -> np.ndarray:
    """
    Compute the Jaccard similarity matrix for two clustering solutions.

    Parameters:
    - Solution1 (list[set]): Clusters of the first solution as sets.
    - Solution2 (list[set]): Clusters of the second solution as sets.

    Returns:
    - MatrixJaccard (np.ndarray): Jaccard similarity matrix for the solutions' clusters.
    """
    try:
        # checks.
        if not isinstance(Solution1, list) or not all(isinstance(s, set) for s in Solution1):
            raise TypeError("Solution1 debe ser una lista de conjuntos (sets).")
        if not isinstance(Solution2, list) or not all(isinstance(s, set) for s in Solution2):
            raise TypeError("Solution2 debe ser una lista de conjuntos (sets).")
        if len(Solution1) == 0 or len(Solution2) == 0:
            raise ValueError("Ninguna de las soluciones debe estar vacía.")

        # Construction of matrix.
        n1 = len(Solution1)
        n2 = len(Solution2)
        MatrixJaccard = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                union = len(Solution1[i] | Solution2[j])
                intersection = len(Solution1[i] & Solution2[j])
                MatrixJaccard[i, j] = 0 if union == 0 else intersection / union

        return MatrixJaccard

    except Exception as e:
        print(f"Error en Jaccar_similarityClusters: {e}")
        return np.array([])

def compute_rand_index(
        labels1: np.ndarray, 
        labels2: np.ndarray, 
        adjusted: bool = True
    ) -> float:
    """
    Compute Rand Index or Adjusted Rand Index between two clustering solutions.
    
    Parameters:
    - labels1 (np.ndarray): First clustering solution.
    - labels2 (np.ndarray): Second clustering solution.
    - adjusted (bool): If True, use Adjusted Rand Index. Otherwise, use Rand Index.
    
    Returns:
    - float: Similarity score.
    """
    if len(labels1) != len(labels2):
        raise ValueError("Both input arrays must have the same length.")
    
    return adjusted_rand_score(labels1, labels2) if adjusted else rand_score(labels1, labels2)

def process_RandValues(
        Matrix: list[np.ndarray], 
        n_threads: int, 
        adjusted: bool = True
    ) -> np.ndarray:
    """
    Compute the pairwise Rand or Adjusted Rand Index matrix for a set of clustering solutions.
    
    Parameters:
    - Matrix (list[np.ndarray] or 2D np.ndarray): List or 2D array of clustering solutions.
    - n_threads (int): Number of threads to use.
    - adjusted (bool): Whether to compute Adjusted Rand Index (True) or regular Rand Index (False).
    
    Returns:
    - np.ndarray: Similarity matrix (symmetric).
    """
    try:
        # Validar tipo y estructura
        if isinstance(Matrix, list):
            if not all(isinstance(row, np.ndarray) for row in Matrix):
                raise TypeError("If Matrix is a list, all elements must be NumPy arrays.")
        elif isinstance(Matrix, np.ndarray):
            if Matrix.ndim != 2:
                raise ValueError("Matrix must be a 2D NumPy array.")
            # Convertimos filas a lista de vectores
            Matrix = [Matrix[i, :] for i in range(Matrix.shape[0])]
        else:
            raise TypeError("Matrix must be a list of NumPy arrays or a 2D NumPy array.")

        if n_threads <= 0:
            raise ValueError("n_threads must be a positive integer.")

        n_solutions = len(Matrix)
        if n_solutions == 0:
            raise ValueError("Matrix must contain at least one solution.")

        # Inicializar matriz de similitud
        Similarity_Matrix = np.zeros((n_solutions, n_solutions))

        # Función auxiliar para el cálculo paralelo
        def compute_pairwise_rand(indices):
            i, j = indices
            score = compute_rand_index(Matrix[i], Matrix[j], adjusted=adjusted)
            return i, j, score

        # Índices del triángulo superior (incluyendo diagonal)
        indices = [(i, j) for i in range(n_solutions) for j in range(i, n_solutions)]

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            results = list(executor.map(compute_pairwise_rand, indices))

        # Rellenar matriz simétrica
        for i, j, value in results:
            Similarity_Matrix[i, j] = value
            Similarity_Matrix[j, i] = value

        return Similarity_Matrix

    except Exception as e:
        print(f"Error in process_RandValues: {e}")
        return np.array([])

def compare_solution_pair(
        idx1: int, 
        idx2: int, 
        solutions: list[list[set]]
    ) -> list[tuple[int, int, float]]:
    """
    Compares two solutions and returns the equivalence pairs.

    Parameters:
    - idx1 (int): Index of the first solution.
    - idx2 (int): Index of the second solution.
    - solutions (list): List of clustering solutions (each is a list of sets).

    Returns:
    - equivalent_pairs (list): List of tuples (cluster_i, cluster_j, similarity).
    """
    try:
        MatrixJaccard = Jaccar_similarityClusters(solutions[idx1], solutions[idx2])
        
        similarity_pairs = [
            (i, j, MatrixJaccard[i, j])
            for i in range(len(solutions[idx1]))
            for j in range(len(solutions[idx2]))
        ]
        
        similarity_pairs.sort(key=lambda x: x[2], reverse=True)

        equivalent_pairs = []
        used_clusters_solution1 = set()
        used_clusters_solution2 = set()

        for i, j, similarity in similarity_pairs:
            if i not in used_clusters_solution1 and j not in used_clusters_solution2:
                equivalent_pairs.append((i, j, similarity))
                used_clusters_solution1.add(i)
                used_clusters_solution2.add(j)

        return equivalent_pairs

    except Exception as e:
        print(f"Error comparing solutions at indices ({idx1}, {idx2}): {e}")
        return []

def find_equivalent_clusters(
        solutions: list[list[set]]
    ) -> pd.DataFrame:
    """
    Encuentra los grupos equivalentes entre varias soluciones de agrupamiento usando Jaccard.

    Parameters:
    - solutions (list of list of sets): Lista de soluciones de clustering.

    Returns:
    - pd.DataFrame: DataFrame con combinaciones de grupos equivalentes y sus similitudes.
    """
    try:
        if not isinstance(solutions, list) or not all(isinstance(sol, list) for sol in solutions):
            raise TypeError("Each solution must be a list of sets.")
        if not all(all(isinstance(cl, set) for cl in sol) for sol in solutions):
            raise TypeError("Each cluster must be a set.")

        all_equivalent_pairs = []

        with ThreadPoolExecutor() as executor:
            future_to_comparison = {
                executor.submit(compare_solution_pair, idx1, idx2, solutions): (idx1, idx2)
                for idx1 in range(len(solutions))
                for idx2 in range(idx1 + 1, len(solutions))
            }

            for future in future_to_comparison:
                try:
                    equivalent_pairs = future.result()
                    idx1, idx2 = future_to_comparison[future]
                    all_equivalent_pairs.append(((idx1, idx2), 
                                                 [(pair[0], pair[1]) for pair in equivalent_pairs],
                                                 [pair[2] for pair in equivalent_pairs]))
                except Exception as e:
                    idx1, idx2 = future_to_comparison[future]
                    print(f"Error processing pair ({idx1}, {idx2}): {e}")
        
        return pd.DataFrame(all_equivalent_pairs, columns=["Solution Pair", "Equivalent Clusters", "Jaccard Similarities"])

    except Exception as e:
        print(f"Error in find_equivalent_clusters: {e}")
        return pd.DataFrame(columns=["Solution Pair", "Equivalent Clusters", "Jaccard Similarities"])