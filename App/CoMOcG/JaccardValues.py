######### Libraries #########
import numpy as np                                                # Efficient Math Operations.
from concurrent.futures import ThreadPoolExecutor, as_completed   # Thread Administration.
import pandas as pd                                               # Dataframe managment.

######### Functions #########

"""
This block contains all main functions.
"""

def JaccardIndexSolutions(Solutions_Matrix: np.ndarray, n_threads: int) -> np.ndarray:
    """
    JaccardIndexSolutions(function): Calculate Jaccard index to compare every solution in parallel.

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
            raise ValueError("Matrix at least needs to have two columns (or genes) for valid comparison.")

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
        similarity_vectors = np.array([same_matrix[upper_tri_indices] for same_matrix in all_same_matrices])

        # Initialize return matrix.
        Jaccard_Matrix = np.zeros((num_rows, num_rows))

        # Calculus of every component of Jaccard Index of clustering solutions:
        # Being A and B clustering solutions:
        # r -> two genes are toghether in A and B.
        # u -> two genes are toghether in A but no in B.
        # v -> two genes are toghether in B but no in A.
        # This is a inner function for ThreadPoolExecutor performance.
        def compute_jaccard(i, j):
            r = np.sum(similarity_vectors[i] & similarity_vectors[j])
            u = np.sum(similarity_vectors[i] & ~similarity_vectors[j])
            v = np.sum(~similarity_vectors[i] & similarity_vectors[j])
            return (i, j, r / (r + u + v) if (r + u + v) > 0 else 0.0)

        # Do 'compute_Jaccard between all solutions'
        pairs = [(i, j) for i in range(num_rows) for j in range(i + 1, num_rows)]
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = {executor.submit(compute_jaccard, i, j): (i, j) for i, j in pairs}

            for future in as_completed(futures):
                i, j, jaccard = future.result()
                Jaccard_Matrix[i, j] = jaccard
                Jaccard_Matrix[j, i] = jaccard

        # Diagonal of 1's.
        np.fill_diagonal(Jaccard_Matrix, 1.0)

    except Exception as e:
        raise RuntimeError(f"Something went wrong: {e}")
    else:
        print("Jaccard Index of your solutions successfully calculated.")
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

def CompareSolutionsPair(
        idx1: int, 
        idx2: int, 
        solutions: list[list[set]]
    ) -> list[tuple[int, int, float]]:
    """
    CompareSolutionsPair(function): Compares two solutions and returns the equivalence pairs.

    Parameters:
    - idx1 (int): Index of the first solution.
    - idx2 (int): Index of the second solution.
    - solutions (list): List of clustering solutions (each is a list of sets).

    Returns:
    - equivalent_pairs (list): List of tuples (cluster_i, cluster_j, similarity).
    """
    try:
        # Generate Jaccard index comparison among clusters of the groups (clusters).
        MatrixJaccard = JaccardIndexClusters(solutions[idx1], solutions[idx2])

        # Sort of Jaccard index obtained.
        similarity_pairs = sorted(
            [(cluster_1, cluster_2, MatrixJaccard[cluster_1, cluster_2]) 
             for cluster_1 in range(len(solutions[idx1])) 
             for cluster_2 in range(len(solutions[idx2]))], 
            key=lambda x: x[2], reverse=True
        )

        # Take just pairs of cluster that have max jaccard index, ensuring no
        # cluster duplication.
        used_clusters_1, used_clusters_2 = set(), set()
        return [(cluster_1, cluster_2, sim) for cluster_1, cluster_2, sim in similarity_pairs if cluster_1 not in used_clusters_1 and cluster_2 not in used_clusters_2 
                and not (used_clusters_1.add(cluster_1) or used_clusters_2.add(cluster_2))]

    except Exception as e:
        raise RuntimeError(f"Error comparing solutions at indices ({idx1}, {idx2}): {e}")

def FindEquivalentClusters(
        solutions: list[list[set]]
    ) -> pd.DataFrame:
    """
    FindEquivalentClusters(function): Identifier all equivalent clusters in solutions collection.

    Parameters:
    - solutions (list of list of sets): List of clustering collection in format providesd by SolutionClusterMatrix function.

    Returns:
    - pd.DataFrame: DataFrame allocating columns - Solution 1,Solution 2,Cluster 1,Cluster 2,Jaccard Similarity - as
      registration of equivalent clusters.
    """
    try:
        # Checking input.
        if not isinstance(solutions, list) or not all(isinstance(sol, list) for sol in solutions):
            raise TypeError("Each solution must be a list of sets.")
        
        # Result structure (rows for dataframe).
        rows = []

        # Concurrent execution.
        with ThreadPoolExecutor() as executor:
            future_to_pair = {
                executor.submit(CompareSolutionsPair, idx1, idx2, solutions): (idx1, idx2)
                for idx1 in range(len(solutions))
                for idx2 in range(idx1 + 1, len(solutions))
            }

            # Define data on rows.
            for future in future_to_pair:
                idx1, idx2 = future_to_pair[future]
                try:
                    equivalent_pairs = future.result()
                    for cluster1_idx, cluster2_idx, similarity in equivalent_pairs:
                        rows.append({
                            "Solution 1": idx1,
                            "Solution 2": idx2,
                            "Cluster 1": cluster1_idx,
                            "Cluster 2": cluster2_idx,
                            "Jaccard Similarity": similarity
                        })
                except Exception as e:
                    raise RuntimeError(f"Error processing pair ({idx1}, {idx2}): {e}")

    except Exception as e:
        raise RuntimeError(f"Error in find_equivalent_clusters: {e}")
    else:
        return pd.DataFrame(rows)