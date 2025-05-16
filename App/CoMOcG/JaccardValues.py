######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.
import pandas as pd                                 # Dataframe managment.

######### Functions #########

"""
This block contains all main functions.
"""

def JaccardIndexSolutions(Solutions_Matrix: np.ndarray):
    """
    JaccardIndexSolutions(function): Calculate Jaccard index to compare every solution.

    Parameters:
        - Solutions_Matrix: Clustering solutions represented by 1D integers Array.
    Returns:
        - Jaccard_Matrix: Jaccard Index of every pair of solutions.
    """
    try:
        # Checking Matrix dimension to ensure if it is not empty.
        if Solutions_Matrix.shape[0] == 0:
            raise ValueError("Empty solutions matrix.")
        elif Solutions_Matrix.shape[1] < 2:
            raise ValueError("Matrix at least needs to have two columns (or genes) for valid comparision in solutions.")

        num_rows = Solutions_Matrix.shape[0]            # Amount solutions.
        n_elements = Solutions_Matrix.shape[1]          # Amount genes.
        
        # Create a reshaped matrix that represents all comparitions between solutions adding
        # a new third dimension.
        solutions_expanded = Solutions_Matrix.reshape(num_rows, n_elements, 1)
        
        # Broadcasting: Compare every array with itself, this tell us if two genes (element or column)
        # are thogether in the clusters that allocates the solution.
        # Avoid clusters labels confusion errors.
        all_same_matrices = (solutions_expanded == Solutions_Matrix.reshape(num_rows, 1, n_elements))
        
        # Unique pairs of genes (no same elements pairs).
        upper_tri_indices = np.triu_indices(n_elements, k=1)
        
        # Take upper triangular index for the comparison, because the index is symetric.
        similarity_vectors = np.array([same_matrix[upper_tri_indices] for same_matrix in all_same_matrices])
        
        # Initialize return matrix.
        Jaccard_Matrix = np.zeros((num_rows, num_rows))
        
        # Calculus of every component of Jaccard Index of clustering solutions:
        # Being A and B clustering solutions:
        # r -> two genes are toghether in A and B.
        # s -> two genes are separated in A and B.
        # u -> two genes are toghether in A but no in B.
        # u -> two genes are toghether in B but no in A.
        for i in range(num_rows):
            for j in range(i+1, num_rows):
                r = np.sum(similarity_vectors[i] & similarity_vectors[j])
                u = np.sum(similarity_vectors[i] & ~similarity_vectors[j])
                v = np.sum(~similarity_vectors[i] & similarity_vectors[j])
                
                # Jaccard Index Formula.
                jaccard = r / (r + u + v) if (r + u + v) > 0 else 0.0
                
                # Symmetric.
                Jaccard_Matrix[i, j] = jaccard
                Jaccard_Matrix[j, i] = jaccard
        
        # Diagonal of 1's.
        np.fill_diagonal(Jaccard_Matrix, 1.0)

    except Exception as e:
        raise RuntimeError(f"Something went wrong: {e}")
    else:
        print("Jaccard Index of your solutions succesfully calculated.")
        return Jaccard_Matrix

def JaccardIndexClusters(
        Solution1: list[set], 
        Solution2: list[set]
        ) -> np.ndarray:
    """
    JaccardIndexClusters(function): Compute Jaccard similarity matrix for two clustering solutions,
    evaluating the clusters inside of them.

    Parameters:
    - Solution1 (list[set]): Clusters of the first solution as list of sets.
    - Solution2 (list[set]): Clusters of the second solution as list of sets.

    Returns:
    - MatrixJaccard (np.ndarray): Jaccard similarity matrix among the solutions clusters.
    """
    try:
        # Check type of input in solution 1 and 2. They must be a list of strings sets, with
        # each one of them are a representation of the gene (symbol, entrezID, others).
        if not isinstance(Solution1, list) or not all(isinstance(s, set) for s in Solution1):
            raise TypeError("Frist parameter (Solution1) must be a list of sets (prefer int and str).")
        elif not isinstance(Solution2, list) or not all(isinstance(s, set) for s in Solution2):
            raise TypeError("Second parameter (Solution2) must be a list of sets (prefer int and str).")
        # Non-empty solution.
        elif len(Solution1) == 0 or len(Solution2) == 0:
            raise ValueError("One of the solutions are empty, no comparison is possible.")
        
        # Construction of matrix.
        n1 = len(Solution1)
        n2 = len(Solution2)
        MatrixJaccard = np.zeros((n1, n2))

        # Iteration to compute Jaccard Index (sets version).
        # We take the set allocated and their respective index.
        MatrixJaccard = np.fromiter(
            (len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0 for s1 in Solution1 for s2 in Solution2),
            dtype=float
        ).reshape(n1, n2)

    except TypeError as te:
        raise RuntimeError(f"Type error in input sets: {te}")
    except Exception as e:
        raise RuntimeError (f"Something went wrong: {e}")
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
        MatrixJaccard = JaccardIndexClusters(solutions[idx1], solutions[idx2])
        
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