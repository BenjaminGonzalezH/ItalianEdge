######### Libraries #########
from collections import defaultdict                 # Dictionary.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.

######### Functions #########

def ProcessSolution_IDs(solution, genes):
    """
    ProcessSolution_IDs(function)
        Input:
            - solution: List of index that indicates the cluster
            where the gene are in.
            - genes: List of genes identificators.
        Output:
            - Clusters: List that allocates list of genes IDs.

        Description: Function that process a solution from
        'ReadInputCSV' that reads a CSV with the following format:
        
        Gene 1;Gene 2;Gene 3;...;Gene n
        Solution 1;1;2;4;1
        Solution 2;1;1;2;4;1
        Solution 3;1;1;2;4;1
        ...;
        Solution k;1;1;2;4;1

        The numbers of this data is transform into a matrix just with
        the numbers and this function just take one and transform
        in this:

        [
            ['GENE1', 'GENE5'], 
            ['GENE2', 'GENE4'], 
            ['GENE3', 'GENE6', 'GENE8'], 
            ['GENE7']
        ]
    """
    try:
        # Create a dictionary that allocates the index with a 
        # determinated value (int). First creates a default that
        # generate a empty list for every new key.
        # Next, allocates the data from the solution.
        Index_values = defaultdict(list)
        for index, value in enumerate(solution):
            Index_values[value].append(index)

        # Create a list of list of genes IDs.
        clusters = []
        for key in Index_values.keys():
            cluster = []
            for value in Index_values[key]:
                cluster.append(genes[value])
            clusters.append(cluster)
        
        # Output.
        return clusters

    # Exception Block.
    except TypeError:
        print("Error: La entrada debe ser una lista de valores numéricos.")
        return None
    except Exception:
        print("Ha ocurrido un error inesperado.")
        return None

def ProcessSolution_noIDs(solution):
    """
    ProcessSolution_noIDs(function)
        Input:
            - solution: List of index that indicates the cluster
            where the gene are in.
        Output:
            - Clusters: List that allocates list of index.

        Description: Function that process a solution from
        'ReadInputCSV' that reads a CSV with the following format:
        
        Gene 1;Gene 2;Gene 3;...;Gene n
        Solution 1;1;2;4;1
        Solution 2;1;1;2;4;1
        Solution 3;1;1;2;4;1
        ...;
        Solution k;1;1;2;4;1

        The numbers of this data is transform into a matrix just with
        the numbers and this function just take one and transform
        in this:

        [
            [0, 4], 
            [1, 3], 
            [2, 5, 7], 
            [6]
        ]
    """
    try:
        # Create a dictionary that allocates the index with a 
        # determinated value (int). First creates a default that
        # generate a empty list for every new key.
        # Next, allocates the data from the solution.
        Index_values = defaultdict(list)
        for index, value in enumerate(solution):
            Index_values[value].append(index)

        # Create a list of list of genes index.
        clusters = []
        for key in Index_values.keys():
            cluster = []
            for value in Index_values[key]:
                cluster.append(value)
            clusters.append(cluster)
        
        # Output.
        return clusters

    # Exception Block.
    except TypeError:
        print("Error: La entrada debe ser una lista de valores numéricos.")
        return None
    except Exception:
        print("Ha ocurrido un error inesperado.")
        return None

def SolutionClusterMatrix_GeneID(Matrix, genes, max_workers=4):
    """
    SolutionClusterMatrix_GeneID(function)
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
        # Using 'ThreadPoolExecutor' to make a parallel processing of
        # solutions. This help us to mantain mutual exclusion and manage
        # the threads.
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
    SolutionClusterMatrix_NoGeneID(function)
        Input:
            -Matrix: Cluster results matrix created
            from the function 'ReadInputCSV'.
        Output:
            -SolutionClusterMatrix: In his versión with
            index of the genes.
        Description: Function to create a gene clustering matrix
        that means the following structure:
        
        Solution/cluster|   Cluster 1     |   Cluster 2   | ...
        Solution 1      |     1, 2,...    |   ...         | ...
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
        # Using 'ThreadPoolExecutor' to make a parallel processing of
        # solutions. This help us to mantain mutual exclusion and manage
        # the threads.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create connectievity matrix of the solution.
            SolutionClusterMatrix = list(executor.map(lambda solution: ProcessSolution_noIDs(solution), Matrix))
        
        # Output.
        return SolutionClusterMatrix
    
    # Exceptions block.
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")