######### Libraries #########
import cupy as cp                                   # Efficient Math Operations (GPU).

######### Functions #########

"""
This block contains all main functions.
"""

def RandIndexSolutions(Solutions_Matrix: cp.ndarray) -> cp.ndarray:
    """
    RandIndexSolutions(function): Compute the pairwise Rand Index or Adjusted Rand Index matrix using vectorization.

    Parameters:
        - Solutions_Matrix: Clustering solutions represented by 1D integers Array.
    Returns:
        - Rand_Matrix: Rand Index of every pair of solutions.
    """
    try:
        if Solutions_Matrix.shape[0] == 0:
            raise ValueError("Empty solutions matrix.")
        elif Solutions_Matrix.shape[1] < 2:
            raise ValueError("Matrix needs at least two columns (genes) for valid comparison.")

        num_rows, n_elements = Solutions_Matrix.shape

        # Expande para comparación tipo broadcasting
        solutions_expanded = Solutions_Matrix.reshape(num_rows, n_elements, 1)
        all_same_matrices = (solutions_expanded == Solutions_Matrix.reshape(num_rows, 1, n_elements))

        # Indices del triángulo superior (pares únicos de genes)
        upper_tri_indices = cp.triu_indices(n_elements, k=1)

        # Vector binario de coincidencias para cada solución
        similarity_vectors = cp.array([same_matrix[upper_tri_indices] for same_matrix in all_same_matrices])

        # Inicializa matriz de resultado (diagonal = 1)
        Rand_Matrix = cp.eye(num_rows, dtype=cp.float64)
        denom = n_elements*(n_elements-1)/2

        # Cálculo vectorizado (GPU) para cada par
        for i in range(num_rows):
            for j in range(i + 1, num_rows):
                r = cp.sum(similarity_vectors[i] & similarity_vectors[j])
                s = cp.sum(~similarity_vectors[i] & ~similarity_vectors[j])
                rand = (r + s) / denom if denom > 0 else 0.0
                Rand_Matrix[i, j] = rand
                Rand_Matrix[j, i] = rand

    except Exception as e:
        raise RuntimeError(f"Something went wrong: {e}")

    print("Rand Index of your solutions successfully calculated (on GPU).")
    return Rand_Matrix