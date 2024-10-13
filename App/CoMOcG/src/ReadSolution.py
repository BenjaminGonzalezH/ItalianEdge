######### Libraries #########
import numpy as np
import csv

######### Error Codes #########


######### Functions #########

def ReadInputcsv(filepath):
    """
        ReadInput (function)
            - filepath: filepath of your Multiobjective
              gene clustering results.
    """
    try:
        # Gene clustering results matrix.
        Matrix = []

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
        return "FNF"
    except OSError:
        return "I/O_E"
    except EOFError:
        return "EOF_E"