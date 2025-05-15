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

def JaccardIndexSolutions(Solutions_Matrix: np.ndarray):
    num_rows = Solutions_Matrix.shape[0]
    n_elements = Solutions_Matrix.shape[1]
    
    # Crear todas las matrices de comparación de una vez
    # Reshape Solutions_Matrix para broadcasting
    solutions_expanded = Solutions_Matrix.reshape(num_rows, n_elements, 1)
    
    # Broadcasting: comparar cada solución con sí misma para obtener matrices de similitud
    # Resultado: un tensor 3D (num_rows, n_elements, n_elements)
    all_same_matrices = (solutions_expanded == Solutions_Matrix.reshape(num_rows, 1, n_elements))
    
    # Índices del triángulo superior (sin diagonal)
    upper_tri_indices = np.triu_indices(n_elements, k=1)
    
    # Extraer los vectores de similitud del triángulo superior para cada solución
    # Shape: (num_rows, número de elementos en el triángulo superior)
    similarity_vectors = np.array([same_matrix[upper_tri_indices] for same_matrix in all_same_matrices])
    
    # Inicializar matriz resultado
    Jaccard_Matrix = np.zeros((num_rows, num_rows))
    
    # Calcular todas las medidas Jaccard de una vez usando broadcasting
    for i in range(num_rows):
        for j in range(i+1, num_rows):
            # Operaciones vectorizadas entre las soluciones i y j
            r = np.sum(similarity_vectors[i] & similarity_vectors[j])
            u = np.sum(similarity_vectors[i] & ~similarity_vectors[j])
            v = np.sum(~similarity_vectors[i] & similarity_vectors[j])
            
            # Calcular índice Jaccard
            jaccard = r / (r + u + v) if (r + u + v) > 0 else 0.0
            
            # Asignar a la matriz (simétrica)
            Jaccard_Matrix[i, j] = jaccard
            Jaccard_Matrix[j, i] = jaccard
    
    # Diagonal a 1
    np.fill_diagonal(Jaccard_Matrix, 1.0)
    
    return Jaccard_Matrix

def JaccarIndexClusters(
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
        elif not isinstance(Solution2, list) or not all(isinstance(s, set) for s in Solution2):
            raise TypeError("Solution2 debe ser una lista de conjuntos (sets).")
        if len(Solution1) == 0 or len(Solution2) == 0:
            raise ValueError("Ninguna de las soluciones debe estar vacía.")

        # Construction of matrix.
        n1 = len(Solution1)
        n2 = len(Solution2)
        MatrixJaccard = np.zeros((n1, n2))

        for i, s1 in enumerate(Solution1):
            for j, s2 in enumerate(Solution2):
                union_size = len(s1 | s2)
                intersection_size = len(s1 & s2)
                MatrixJaccard[i, j] = intersection_size / union_size if union_size != 0 else 0

    except Exception as e:
        raise RuntimeError (f"Error en Jaccar_similarityClusters: {e}")
    else:
        return MatrixJaccard

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
        MatrixJaccard = JaccarIndexClusters(solutions[idx1], solutions[idx2])
        
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