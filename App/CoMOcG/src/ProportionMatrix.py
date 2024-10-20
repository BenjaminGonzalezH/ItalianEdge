######### Libraries #########
import itertools            # Eficient iterations.
import os                   # OS callings.
import numpy as np          # Math and Structures.
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.

######### Functions #########

def compute_connectivity(solution):
    n = len(solution)
    aux_matrix = lil_matrix((n, n))  # Crear una matriz dispersa de tipo LIL (Linked List).
    
    for i in range(n):
        for j in range(i + 1, n):
            if solution[i] == solution[j]:
                aux_matrix[i, j] = 1
                aux_matrix[j, i] = 1
    
    return aux_matrix.tocsr()

def connectivityMatrix(SolutionsMatrix, max_workers=4):
    try:
        # Usamos un ThreadPoolExecutor para procesar soluciones en paralelo
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Ejecutamos el cálculo en paralelo para cada solución
            connectivityMatrix = list(executor.map(compute_connectivity, SolutionsMatrix))
        
        return connectivityMatrix
    
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

def sum_connectivity_matrices(connectivity_matrices):
    """
    Suma todas las matrices de conectividad y retorna una sola matriz dispersa (csr_matrix).
    :param connectivity_matrices: Lista de matrices de conectividad en formato CSR.
    :return: Una matriz CSR que es la suma de todas las matrices de conectividad.
    """
    if not connectivity_matrices:
        return None  # Manejo de caso cuando no hay matrices para sumar.

    # Inicializamos la suma con una matriz CSR de ceros del mismo tamaño que la primera matriz.
    summed_matrix = csr_matrix(connectivity_matrices[0].shape)

    # Sumamos todas las matrices de conectividad.
    for matrix in connectivity_matrices:
        summed_matrix += matrix

    return summed_matrix

def divide_by_total_solutions(summed_matrix, total_solutions):
    """
    Divide cada valor de la matriz sumada de conectividad por el total de soluciones.
    
    :param summed_matrix: Matriz sumada de conectividad (csr_matrix).
    :param total_solutions: Número total de soluciones.
    :return: Matriz con valores divididos por el total de soluciones.
    """
    # Asegurarse de no dividir por cero.
    if total_solutions == 0:
        raise ValueError("El número total de soluciones no puede ser cero.")

    # Realizar la división en la matriz dispersa.
    divided_matrix = summed_matrix / total_solutions
    
    return divided_matrix


def subtract_from_one(matrix):
    """
    Resta cada valor de la matriz a 1.
    
    :param matrix: Matriz de conectividad con valores entre 0 y 1 (csr_matrix).
    :return: Matriz con cada valor restado a 1.
    """
    # Convertir a matriz densa, restar a 1, y luego volver a convertir a matriz dispersa si se necesita.
    dense_matrix = matrix.toarray()  # Convertimos la matriz dispersa a formato denso.
    subtracted_matrix = 1 - dense_matrix  # Realizamos la resta a 1.
    
    # Si prefieres mantener la matriz en formato disperso, puedes convertirla de nuevo a csr_matrix.
    return csr_matrix(subtracted_matrix)