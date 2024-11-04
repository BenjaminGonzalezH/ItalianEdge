######### Libraries #########
from scipy.sparse import lil_matrix                 # List of list matrix.
from scipy.sparse import csr_matrix                 # Compresed space row matrix.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.

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
        Proportion_matrix = summed_matrix / total_solutions
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
        dense_matrix = matrix.toarray()
        subtracted_matrix = 1 - dense_matrix
        return csr_matrix(subtracted_matrix)
    
    # Exceptions block.
    except AttributeError as e:
        print(f"Error: la matriz proporcionada no es válida o no tiene el método 'toarray'. Detalles: {e}")
    except TypeError as e:
        print(f"Error: tipo de datos incorrecto en la matriz. Detalles: {e}")
    except Exception as e:
        print(f"Ha ocurrido un error inesperado. Detalles: {e}")
