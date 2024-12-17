######### Libraries #########
import numpy as np
import os
import itertools
from scipy.sparse import csr_matrix                 # Compresed space row matrix.
from concurrent.futures import ProcessPoolExecutor  # Process Administration.
from concurrent.futures import ThreadPoolExecutor   # Threads Administration.


def compute_proportion_genesolution(Solution1, Solution2):
    """
    """
    # Convertir a numpy arrays para optimización
    Solution1 = np.array(Solution1)
    Solution2 = np.array(Solution2)
    
    # Validaciones
    if len(Solution1) == 0 or len(Solution2) == 0:
        raise ValueError("Las soluciones no pueden estar vacías.")
    if len(Solution1) != len(Solution2):
        raise ValueError("Las soluciones deben tener la misma longitud.")
    
    # Calcular la proporción
    count = np.sum(Solution1 == Solution2)  # Suma de coincidencias
    proportion = count / len(Solution1)

    return proportion

def process_proportion_genessolution(SolutionsMatrix, n_threads):
    """
    Calcula la matriz de conectividad entre todas las soluciones en paralelo.
    
    Parameters:
    SolutionsMatrix (list or np.ndarray): Matriz de soluciones (listas o NumPy).
    n_threads (int): Número de hilos para paralelizar el cálculo.
    
    Returns:
    np.ndarray: Matriz de conectividad con proporciones de coincidencias.
    """
    # Verificaciones de entrada
    if isinstance(SolutionsMatrix, list):
        if not all(isinstance(row, list) for row in SolutionsMatrix):
            raise ValueError("SolutionsMatrix debe ser una lista de listas.")
    elif isinstance(SolutionsMatrix, np.ndarray):
        if SolutionsMatrix.ndim != 2:
            raise ValueError("SolutionsMatrix debe ser un arreglo NumPy 2D.")
    else:
        raise ValueError("SolutionsMatrix debe ser una lista de listas o un arreglo NumPy 2D.")
    
    if n_threads <= 0:
        raise ValueError("n_threads must be a positive integer.")
    
    # Convertir SolutionsMatrix a un arreglo NumPy para facilitar el manejo
    SolutionsMatrix = np.array(SolutionsMatrix)

    # Crear la matriz de conectividad vacía
    n_solutions = len(SolutionsMatrix)
    connectivityMatrix = np.zeros((n_solutions, n_solutions))

    # Definir función auxiliar para paralelizar
    def compute_pairwise_proportion(indices):
        i, j = indices
        return i, j, compute_proportion_genesolution(SolutionsMatrix[i], SolutionsMatrix[j])
    
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