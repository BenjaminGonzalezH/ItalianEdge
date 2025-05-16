######### Libraries #########
import numpy as np                                      # Efficient Math Operations.

######### Functions #########

"""
This block contains all main functions.
"""

def RandIndexSolutions(Solutions_Matrix: np.ndarray) -> np.ndarray:
    """
    RandIndexSolutions(function): Compute the pairwise Rand Index or Adjusted Rand Index matrix using vectorization.

    Parameters:
        - Solutions_Matrix: Clustering solutions represented by 1D integers Array.
    Returns:
        - Rand_Matrix: Rand Index of every pair of solutions.
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
        Rand_Matrix = np.zeros((num_rows, num_rows))
        
        # Calculus of every component of Jaccard Index of clustering solutions:
        # Being A and B clustering solutions:
        # r -> two genes are toghether in A and B.
        # s -> two genes are separated in A and B.
        for i in range(num_rows):
            for j in range(i+1, num_rows):
                r = np.sum(similarity_vectors[i] & similarity_vectors[j])
                s = np.sum(~similarity_vectors[i] & ~similarity_vectors[j])
                
                # Rand Index Formula.
                Rand_index = (r+s) / (n_elements*(n_elements-1)/2) if (n_elements*(n_elements-1)/2) > 0 else 0.0
                
                # Symmetric.
                Rand_Matrix[i, j] = Rand_index
                Rand_Matrix[j, i] = Rand_index
        
        # Diagonal of 1's.
        np.fill_diagonal(Rand_Matrix, 1.0)

    except Exception as e:
        raise RuntimeError(f"Something went wrong: {e}")
    else:
        print("Rand Index of your solutions succesfully calculated.")
        return Rand_Matrix