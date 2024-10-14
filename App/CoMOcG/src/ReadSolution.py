######### Libraries #########
import os
import numpy as np
import csv

######### Functions #########

def ReadInputcsv(filepath):
    """
    ReadInput (function)
        Input:
            - filepath: filepath of your Multiobjective
              gene clustering results.
        Output:
            - Matrix: Integer values matrix thar allocates
            numbers of clusters associated with every gene.
            - gene: List of the genes used in the study.
            - n: amount of genes.

        Description: This function reads a .csv file with the
        following format.
                    Gene 1 | Gene 2 | Gene 3 | ... | Gene n
        Solution 1|    1        3       4               2
        Solution 2|    3        4       1               2
        Solution 3|
        ...
        Solution k|

        Every cell in the "dataframe" allocates the number
        of the cluster where the gene is in the actual solution.
    """
    try:
        # Gene clustering results matrix.
        Matrix = []

        # Empty file verification.
        if os.path.getsize(filepath) == 0:
            raise ValueError("Empty file.")
        
        # Open CSV file.
        with open(filepath, mode = "r") as file:
            # Start reading process.
            Csvfile = csv.reader(file)

            # List of genes.
            genes = next(Csvfile)
            n = len(genes)

            # Read every line.
            for line in Csvfile:        
                # Take just int values.
                Matrix.append(line[1:n+1])

        return np.array(Matrix, dtype=int), genes, n
    
    except FileNotFoundError:
        raise ValueError(f"File in {filepath} does not exists.")
    except OSError:
        raise ValueError("Can not read the file due to a I/O error.")
    
def ReadInputcsv_noID(filepath):
    """
    ReadInput (function)
        Input:
            - filepath: filepath of your Multiobjective
              gene clustering results.
        Output:
            - Matrix: Integer values matrix thar allocates
            numbers of clusters associated with every gene.
            - gene: List of the genes used in the study.
            - n: amount of genes.

        Description: This function reads a .csv file with the
        following format.
        Gene 1 | Gene 2 | Gene 3 | ... | Gene n
            1        3       4               2
            3        4       1               2  

        Every cell in the "dataframe" allocates the number
        of the cluster where the gene is in the actual solution.
    """
    try:
        # Gene clustering results matrix.
        Matrix = []

        # Empty file verification.
        if os.path.getsize(filepath) == 0:
            raise ValueError("Empty file.")
        
        # Open CSV file.
        with open(filepath, mode = "r") as file:
            # Start reading process.
            Csvfile = csv.reader(file)

            # List of genes.
            genes = next(Csvfile)
            n = len(genes)

            # Read every line.
            for line in Csvfile:        
                # Take just int values.
                Matrix.append(line[0:n+1])

        return np.array(Matrix, dtype=int), genes, n
    
    except FileNotFoundError:
        raise ValueError(f"File in {filepath} does not exists.")
    except OSError:
        raise ValueError("Can not read the file due to a I/O error.")
