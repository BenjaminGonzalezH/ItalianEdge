######### Libraries #########
from collections import defaultdict # Dictionary.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.

######### Functions #########

def ProcessSolution_IDs(solution, genes):
    """"""
    try:
        Index_values = defaultdict(list)

        for index, value in enumerate(solution):
            Index_values[value].append(index)

        clusters = []
        for key in Index_values.keys():
            cluster = []
            for value in Index_values[key]:
                cluster.append(genes[value])
            clusters.append(cluster)
        
        return clusters

    except TypeError as e:
        print("Error: La entrada debe ser una lista de valores numéricos.")
        print(f"Detalles del error: {e}")

    except Exception as e:
        print("Ha ocurrido un error inesperado.")
        print(f"Detalles del error: {e}")

def ProcessSolution_noIDs(solution):
    """"""
    try:
        Index_values = defaultdict(list)

        for index, value in enumerate(solution):
            Index_values[value].append(index)

        clusters = []
        for key in Index_values.keys():
            cluster = []
            for value in Index_values[key]:
                cluster.append(value)
            clusters.append(cluster)
        
        return clusters

    except TypeError as e:
        print("Error: La entrada debe ser una lista de valores numéricos.")
        print(f"Detalles del error: {e}")

    except Exception as e:
        print("Ha ocurrido un error inesperado.")
        print(f"Detalles del error: {e}")

def SolutionClusterMatrix_GeneID(Matrix, genes, max_workers=4):
    """
    SolutionClusterMatrix(function)
        Input:
            -Matrix: Cluster results matrix created
            from the function 'ReadInputCSV'.
            -genes: List of genes from the same matrix
            of 'ReadInputCSV'.
        Output:
            -SolutionClusterMatrix: In his versión with
            names of the genes.
        Description: Function to create a gene clustering matrix
        that means the following structure:
        
        Solution/cluster|   Cluster 1           |   Cluster 2   | ...
        Solution 1      | <name:1>, <name_2>,...|   ...         | ...
        ...

        This from two separate structures from data in this format:

        Gene 1;Gene 2;Gene 3;...;Gene n
        1;2;4;1
        1;1;2;4;1
        1;1;2;4;1
        ...;
        1;1;2;4;1
    """
    try:  
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create connectievity matrix of the solution.
            SolutionClusterMatrix = list(executor.map(lambda solution: ProcessSolution_IDs(solution, genes), Matrix))
        
        # Output.
        return SolutionClusterMatrix
    
    # Exceptions block.
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

def SolutionClusterMatrix_NoGeneID(Matrix, max_workers=4):
    """
    SolutionClusterMatrix(function)
        Input:
            -Matrix: Cluster results matrix created
            from the function 'ReadInputCSV'.
            -genes: List of genes from the same matrix
            of 'ReadInputCSV'.
        Output:
            -SolutionClusterMatrix: In his versión with
            names of the genes.
        Description: Function to create a gene clustering matrix
        that means the following structure:
        
        Solution/cluster|   Cluster 1           |   Cluster 2   | ...
        Solution 1      | <name:1>, <name_2>,...|   ...         | ...
        ...

        This from two separate structures from data in this format:

        Gene 1;Gene 2;Gene 3;...;Gene n
        1;2;4;1
        1;1;2;4;1
        1;1;2;4;1
        ...;
        1;1;2;4;1
    """
    try:  
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create connectievity matrix of the solution.
            SolutionClusterMatrix = list(executor.map(lambda solution: ProcessSolution_noIDs(solution), Matrix))
        
        # Output.
        return SolutionClusterMatrix
    
    # Exceptions block.
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")