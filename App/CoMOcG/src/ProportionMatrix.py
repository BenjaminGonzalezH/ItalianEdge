######### Libraries #########
import numpy as np
from scipy.sparse import lil_matrix                 # List of list matrix.
from scipy.sparse import csr_matrix                 # Compresed space row matrix.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.
from itertools import combinations

######### Functions #########

def compute_connectivity(solution):
    """
    compute_connectivity(function)
        Input:
            - Solution: a list that allocates the number of 
            cluster where the gene is in.
        Output:
            - connectivity Matrix: Matrix that indicates pair of
            genes in the same cluster.
        
        Description: Procedure where the threads are going to
        process a solution from the solutions of next function
        checking index with same value on the list.
    """
    try:
        # Check correct type of input data (list or equivalent).
        if not isinstance(solution, (list, tuple)):
            raise TypeError("La solución debe ser una lista o una secuencia.")
        
        # Check not empty list.
        if len(solution) == 0:
            raise ValueError("La solución no puede estar vacía.")
        
        # Initialize length of solution and connectivity matrix as lil_matrix
        n = len(solution)
        aux_matrix = lil_matrix((n, n))
    
        # Check match of every pair of genes.
        for i in range(n):
            for j in range(i, n):
                # It is symetric.
                if solution[i] == solution[j]:
                    aux_matrix[i, j] = 1
                    aux_matrix[j, i] = 1
    
        # Transform to csr to perform matrix
        # operations more efficiently.
        return aux_matrix.tocsr()
    
    # Exceptions block.
    except TypeError as te:
        print(f"Error de tipo: {te}")
        return None
    except ValueError as ve:
        print(f"Error de valor: {ve}")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None    

def connectivityMatrix(SolutionsMatrix, max_workers=4):
    """
    connectivityMatrix(function)
        Input: 
            - SolutionMatrix: Result from functions of reading input files
            (allocates in 'ReadSolution.py' of this package).
            - max_workers: Maximum of threads that are gonna process the
            solutions 1 by 1.
        Output:
            - connectivityMatrix: List of sparse matrix that indicates
            pair of gene who are in the same cluster in the respective
            solution.
        Description: Create a Matrix that create a square matrix with
        max length dimesion from his solution (all equals). This says
        what pair of index share the same number. Imagine the 
        following example:
           Solution 1 = [1, 2, 1, 2]

           Connectivity Matrix = [ [0,1,0,0] 
                                   [1,0,0,1]
                                   [0,0,1,0] ]
        That is for every solution in solution matrix.
    """
    try:
        # Using 'ThreadPoolExecutor' to make a parallel processing of
        # solutions. This help us to mantain mutual exclusion and manage
        # the threads. 
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create connectievity matrix of the solution.
            connectivityMatrix = list(executor.map(compute_connectivity, SolutionsMatrix))
        
        # Output.
        return connectivityMatrix
    
    # Exceptions block.
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

def sum_connectivity_matrices(connectivity_matrix):
    """
    sum_connectivity_matrices(funcion)
        Input:
            - connectivity_matrix: List of sparce matrix from
            'connectivityMatrix' function (csr format).
        Output:
            - summed_matrix: Sum of all connectivity matrix
            from input (csr format).
        Description: Sum all connectivity matrix in the input 
        ('connectivity_matrix' for more information).
    """
    try:
        # Check empty matrix.
        if not connectivity_matrix:
            raise ValueError("La lista de matrices de conectividad está vacía.")

        # Check homogeneus list..
        for matrix in connectivity_matrix:
            if not isinstance(matrix, csr_matrix):
                raise TypeError("Todos los elementos deben ser matrices dispersas CSR.")

        # Initialize matrix thar allocates the sum.
        summed_matrix = csr_matrix(connectivity_matrix[0].shape)

        # Sum.
        for matrix in connectivity_matrix:
            summed_matrix += matrix

        # Output
        return summed_matrix
    
    # Exceptions block.
    except TypeError as te:
        print(f"Error de tipo: {te}")
        return None
    except ValueError as ve:
        print(f"Error de valor: {ve}")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None    

def ProportionMatrix_Similarity(summed_matrix, total_solutions):
    """
    ProportionMatrix_Similarity(function)
        Input:
            - summed_matrix: sum of a list of sparce matrix in 
            csr format.
            - total_solutionns: Amount of solutions to study.
        Output:
            - Proportion_matrix: Matrix that represent the proportion
            of solutions where a pair of genes are allocates in the
            same cluster.
        Description: Proportion Matrix tell us % of cases where a pair
        of index (genes) have the same number (are in the same cluster).
    """
    try:
        # Check CSR format.
        if not isinstance(summed_matrix, csr_matrix):
            raise TypeError("La matriz sumada debe ser del tipo csr_matrix.")
        
        # No division by 0.
        if total_solutions == 0:
            raise ValueError("El número total de soluciones no puede ser cero.")

        # Output.
        Proportion_matrix = summed_matrix.toarray() / total_solutions
        return Proportion_matrix

    # Exceptions block.
    except TypeError as te:
        print(f"Error de tipo: {te}")
        return None
    except ValueError as ve:
        print(f"Error de valor: {ve}")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None    

def ProportionMatrix_Disimilarity(matrix):
    """
    ProportionMatrix_Disimilarity(function)
        Input:
            -matrix: csr matrix.
        Output:
            -subtracted_matrix: matrix with every coeficent
            substrating to 1.
        Description: Proportion Matrix in his similatirie version
        (between 1 and 0).
    """
    try:
        # Output.
        dense_matrix = matrix
        subtracted_matrix = 1 - dense_matrix
        return subtracted_matrix
    
    # Exceptions block.
    except AttributeError as e:
        print(f"Error: la matriz proporcionada no es válida o no tiene el método 'toarray'. Detalles: {e}")
    except TypeError as e:
        print(f"Error: tipo de datos incorrecto en la matriz. Detalles: {e}")
    except Exception as e:
        print(f"Ha ocurrido un error inesperado. Detalles: {e}")


def connectivity_matrices_function(input_matrix, cant_row, cant_col):
    # Diccionario para almacenar matrices de conectividad en formato CSR
    connectivity_matrices = {}

    id_genes = range(cant_col)

    # Iterar sobre cada solución
    for solution_idx in range(0, cant_row):
        print(solution_idx)
        # Crear una lista de tuplas para construir la matriz de conectividad de manera eficiente
        rows, cols, data = [], [], []
        
        # Generar todas las combinaciones de pares de genes
        gene_pairs = list(combinations(id_genes, 2))
        
        # Para cada par de genes, verificar si están conectados
        for first_gene, second_gene in gene_pairs:
            # Si los valores son iguales en ambos genes, existe una conexión
            if input_matrix[solution_idx, first_gene] == input_matrix[solution_idx, second_gene]:
                rows.append(first_gene)
                cols.append(second_gene)
                data.append(1)
        
        # Crear matriz dispersa en formato CSR con los valores recolectados
        connectivity_matrix = csr_matrix((data, (rows, cols)), shape=(cant_col, cant_col))
        
        # Guardar la matriz en el diccionario
        connectivity_matrices[solution_idx] = connectivity_matrix
    
    return connectivity_matrices


def consensus_matrix_function(connectivity_matrices_result, cant_col, cant_row):
    # Crear matrices de ceros en formato denso para almacenar la suma de conexiones
    sum_connectivity_num = np.zeros((cant_col, cant_col))
    
    # Crear la matriz de denominador, donde solo el triángulo superior tiene unos
    connectivity_denom = np.triu(np.ones((cant_col, cant_col), dtype=int), k=1)
    
    # Matriz para almacenar el denominador de la matriz de consenso
    sum_connectivity_denom = np.zeros((cant_col, cant_col))
    
    # Iterar sobre cada fila y columna para acumular las conexiones
    for i in range(cant_col):
        for j in range(cant_col):
            for con_matrix_key in range(cant_row):
                # Extraer el valor actual de la matriz de conectividad en csr_matrix
                aux = connectivity_matrices_result[con_matrix_key][i, j] if (i < cant_col and j < cant_col) else 0
                
                # Sumar el valor en las coordenadas correspondientes en la matriz de numerador
                sum_connectivity_num[i, j] += aux
                sum_connectivity_num[j, i] += aux
                
                # Sumar el valor de la matriz de denominador
                sum_connectivity_denom[i, j] += connectivity_denom[i, j]
                sum_connectivity_denom[j, i] += connectivity_denom[i, j]

    # Calcular la matriz de consenso dividiendo el numerador por el denominador
    with np.errstate(divide='ignore', invalid='ignore'):
        consensus_matrix = np.divide(sum_connectivity_num, sum_connectivity_denom)
    
    # Reemplazar NaN por 0 en la matriz de consenso final
    consensus_matrix = np.nan_to_num(consensus_matrix)
    
    return csr_matrix(consensus_matrix)