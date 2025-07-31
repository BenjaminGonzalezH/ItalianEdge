######### Libraries #########
import cupy as cp                                   # Efficient Math Operations (GPU).

######### Functions #########

"""
This block contains all main functions.
"""

def JaccardIndexSolutions(Solutions_Matrix: cp.ndarray) -> cp.ndarray:
    """
    JaccardIndexSolutions(function): Calculate Jaccard index to compare every solution in parallel.

    Parameters:
        - Solutions_Matrix: Clustering solutions represented by 1D integers Array.
    Returns:
        - Jaccard_Matrix: Jaccard Index of every pair of solutions.
    """
    try:
        # Validaciones básicas.
        if Solutions_Matrix.shape[0] == 0:
            raise ValueError("Empty solutions matrix.")
        elif Solutions_Matrix.shape[1] < 2:
            raise ValueError("Matrix needs at least two columns (genes) for valid comparison.")

        num_rows = Solutions_Matrix.shape[0]
        n_elements = Solutions_Matrix.shape[1]

        # Preparar y expandir matrices para difusión (broadcasting)
        solutions_expanded = Solutions_Matrix.reshape(num_rows, n_elements, 1)
        all_same_matrices = (solutions_expanded == Solutions_Matrix.reshape(num_rows, 1, n_elements))

        upper_tri_indices = cp.triu_indices(n_elements, k=1)

        # Vector binario para cada solución (similitud por pares)
        similarity_vectors = cp.array([same_matrix[upper_tri_indices] for same_matrix in all_same_matrices])

        # Calculo vectorizado por GPU
        Jaccard_Matrix = cp.eye(num_rows, dtype=cp.float64)  # Diagonal de 1´s por definición

        for i in range(num_rows):
            for j in range(i + 1, num_rows):
                r = cp.sum(similarity_vectors[i] & similarity_vectors[j])
                u = cp.sum(similarity_vectors[i] & (~similarity_vectors[j]))
                v = cp.sum((~similarity_vectors[i]) & similarity_vectors[j])
                denom = r + u + v
                jaccard = (r / denom) if denom > 0 else 0.0
                Jaccard_Matrix[i, j] = jaccard
                Jaccard_Matrix[j, i] = jaccard

    except Exception as e:
        raise RuntimeError(f"Something went wrong: {e}")

    else:
        print("Jaccard Index of your solutions successfully calculated (on GPU).")
        return Jaccard_Matrix