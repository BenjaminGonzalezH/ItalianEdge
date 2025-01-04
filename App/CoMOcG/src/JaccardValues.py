######### Libraries #########
import numpy as np
from scipy.sparse import csr_matrix                 # Compresed space row matrix.
from concurrent.futures import ThreadPoolExecutor   # Threads Administration.

def compute_jaccard(Solution1, Solution2) -> csr_matrix:
    """
    compute_jaccard(function): Compute Jaccard D
    """
    # Convertir a arrays de NumPy
    vector1 = np.array(Solution1)
    vector2 = np.array(Solution2)
    
    # Validar longitud
    if len(vector1) != len(vector2):
        raise ValueError("Ambos vectores deben tener la misma longitud.")
    
    # Cálculo del índice de Jaccard
    intersection = np.sum(np.minimum(vector1, vector2))  # Suma de mínimos por elemento
    union = np.sum(np.maximum(vector1, vector2))        # Suma de máximos por elemento
    
    if union == 0:
        return 0.0  # Manejo del caso en que no hay unión
    
    return intersection / union

def process_JaccardValues(SolutionClusterMatrix, n_threads):
    """
    """
    # Verificaciones de entrada
    if isinstance(SolutionClusterMatrix, list):
        if not all(isinstance(row, list) for row in SolutionClusterMatrix):
            raise ValueError("SolutionClusterMatrix debe ser una lista de listas.")
    elif isinstance(SolutionClusterMatrix, np.ndarray):
        if SolutionClusterMatrix.ndim != 2:
            raise ValueError("SolutionClusterMatrix debe ser un arreglo NumPy 2D.")
    else:
        raise ValueError("SolutionClusterMatrix debe ser una lista de listas o un arreglo NumPy 2D.")
    
    if n_threads <= 0:
        raise ValueError("n_threads must be a positive integer.")
    
    # Convertir SolutionClusterMatrix a un arreglo NumPy para facilitar el manejo
    SolutionClusterMatrix = np.array(SolutionClusterMatrix)

    # Crear la matriz de conectividad vacía
    n_solutions = len(SolutionClusterMatrix)
    connectivityMatrix = np.zeros((n_solutions, n_solutions))

    # Definir función auxiliar para paralelizar
    def compute_pairwise_proportion(indices):
        i, j = indices
        return i, j, compute_jaccard(SolutionClusterMatrix[i], SolutionClusterMatrix[j])
    
    # Crear combinaciones de índices
    indices = [(i, j) for i in range(n_solutions) for j in range(i, n_solutions)]
    
    # Procesar en paralelo
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(compute_pairwise_proportion, indices))
    
    # Rellenar la matriz de conectividad
    for i, j, proportion in results:
        connectivityMatrix[i, j] = proportion
        connectivityMatrix[j, i] = proportion  # Simétrica
    
    return connectivityMatrix

def Jaccar_similarityClusters(Solution1, Solution2):

    n = len(Solution1)
    MatrixJaccard = np.zeros((n,n))

    for i in range(len(Solution1)):
        for j in range(i,len(Solution2)):
            union = len(Solution1[i] | Solution2[j])
            intersection = len(Solution1[i] & Solution2[j])
            if union == 0:
                MatrixJaccard[i, j] = 0
                MatrixJaccard[j, i] = 0
            else:
                Jaccard = intersection/union
                MatrixJaccard[i, j] = Jaccard
                MatrixJaccard[j, i] = Jaccard

    return MatrixJaccard

def AmountGenes_Equals(Solution1, Solution2):

    n = len(Solution1)
    MatrixJaccard = np.zeros((n,n))

    for i in range(len(Solution1)):
        for j in range(i,len(Solution2)):
            intersection = len(Solution1[i] & Solution2[j])
            
            MatrixJaccard[i, j] = intersection
            MatrixJaccard[j, i] = intersection

    return MatrixJaccard