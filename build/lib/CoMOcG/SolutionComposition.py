######### Libraries #########
import numpy as np                                  # Efficient Math Operations.

######### Functions #########

"""
This block contains all main functions.
"""

def AmountGenes_Equals(
        Solution1: list[set], 
        Solution2: list[set]
        ) -> np.ndarray:
    """
    AmountGenes_Equals(function): 
    Compute the matrix of shared gene counts between clusters.

    Parameters:
    - Solution1 (list[set]): Clusters of the first solution as sets.
    - Solution2 (list[set]): Clusters of the second solution as sets.

    Returns:
    - MatrixEquals (np.ndarray): Matrix representing the number of shared genes between clusters.
    """
    try:
        # chack inputs.
        if not isinstance(Solution1, list) or not isinstance(Solution2, list):
            raise TypeError("Ambas soluciones deben ser listas de conjuntos (sets).")
        if not all(isinstance(cluster, set) for cluster in Solution1):
            raise TypeError("Todos los elementos de Solution1 deben ser conjuntos (sets).")
        if not all(isinstance(cluster, set) for cluster in Solution2):
            raise TypeError("Todos los elementos de Solution2 deben ser conjuntos (sets).")
        if len(Solution1) == 0 or len(Solution2) == 0:
            raise ValueError("Las soluciones no pueden estar vacías.")

        # Create matrix.
        n1 = len(Solution1)
        n2 = len(Solution2)
        MatrixEquals = np.zeros((n1, n2))

        # Compare sets.
        for i in range(n1):
            for j in range(n2):
                intersection = len(Solution1[i] & Solution2[j])
                MatrixEquals[i, j] = intersection

        return MatrixEquals

    except Exception as e:
        print(f"Error en AmountGenes_Equals: {e}")
        return np.array([])