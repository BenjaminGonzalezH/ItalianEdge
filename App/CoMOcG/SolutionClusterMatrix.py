######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
from collections import defaultdict                 # Dictionary.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.
import pandas as pd

######### Functions #########

"""
This block contains all main functions.
"""

def ProcessSolution(
        solution: np.ndarray, 
        genes: list) -> list:
    """
    ProcessSolution_IDs(function): Function that process a solution from
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
        {'GENE1', 'GENE5'}, 
        {'GENE2', 'GENE4'}, 
        {'GENE3', 'GENE6', 'GENE8'}, 
        {'GENE7'}
    ]

    Parameters:
    - solution (ndarray): List of index that indicates the cluster
    where the gene are in.
    - genes (list): List of genes identificators.
        
    Returns:
    - Clusters (list): List that allocates list of genes IDs.
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
            cluster = set()
            for value in Index_values[key]:
                cluster.add(genes[value])
            clusters.append(cluster)
        
        # Output.
        return clusters

    # Exception Block.
    except TypeError:
        print("Error: Input has to be a list of integers.")
        return None
    except Exception:
        print("Unexpected error.")
        return None

def SolutionClusterMatrix(
        Matrix: list[np.ndarray], 
        genes: list, 
        max_workers:int= 4) -> list:
    """
    SolutionClusterMatrix_GeneID(function): Function to create a gene clustering matrix
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

    Parameters:
    - Matrix (list[np.ndarray]): Cluster results matrix created from the 
    function 'ReadInputCSV'.
    - genes (list): List of genes from the same matrix of 'ReadInputCSV'.
        
    Returns:
    - SolutionClusterMatrix (list[]): In his versión with names of the genes.
    """
    try:
        # Using 'ThreadPoolExecutor' to make a parallel processing of
        # solutions. This help us to mantain mutual exclusion and manage
        # the threads.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create connectievity matrix of the solution.
            SolutionClusterMatrix = list(executor.map(lambda solution: ProcessSolution(solution, genes), Matrix))
        
        # Output.
        return SolutionClusterMatrix
    
    # Exceptions block.
    except Exception as e:
        print(f"Unexpected error: {e}")

def process_solution_vectorized(solution: np.ndarray, genes: list) -> list:
    """
    Versión vectorizada de ProcessSolution:
    Agrupa genes por etiquetas de cluster indicadas en solution.
    Retorna lista de sets de genes por cluster.
    """
    unique_clusters, inverse_indices = np.unique(solution, return_inverse=True)
    # unique_clusters: etiquetas únicas ordenadas
    # inverse_indices: para cada posición, índice en unique_clusters

    clusters = []
    for cluster_label in unique_clusters:
        # Índices donde solution == cluster_label
        indices = np.where(solution == cluster_label)[0]
        # Genes correspondientes
        cluster_genes = set(genes[i] for i in indices)
        clusters.append(cluster_genes)
    return clusters