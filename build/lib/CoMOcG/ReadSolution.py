######### Libraries #########
import itertools                                    # Eficient iterations.
import os                                           # OS callings.
import numpy as np                                  # Efficient Math Operations.
import csv                                          # Read csv.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.
import mmap                                         # Mapping data in memory.
from typing import Tuple                            # Multiple returns doc.

# Note: the return is not a Tuple, this is used for
# allocates all the multiple elements of output that
# would have one function.

######### Functions #########

"""
This block contains all main functions.
"""

def read_csv_part(
        filepath: str, 
        start_row: int, 
        chunk_size: int, 
        flag_solutions_id = 0) -> list[list[int]]:
    """
    read_csv_part(function): This function is use inside of ReadInputCSV for
        multiple process that creates for input reading.

    Parameters:
    - filepath (str): Path that allocates the file in the computer.
    - start_row (int) Row where the thread is going to start reading the file.
    - chunk_size (int): Amount of rows that thread is gonna read.
    - flag_id (int): Flag that indicates if the CSV file uses ID for indicate order in his solutions.
                - 0: There is solutions ID.
                - 1: Opposite of 0.

    Returns: 
    - list_rows (list): All the rows readed.
    """
    # Exception handling: This allows the program to handle common
    # errors in this task.
    try:
        # Validate inputs.
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if start_row < 0:
            raise ValueError("start_row must be a non-negative integer.")
        
        # Output variable.
        list_rows = []

        # with: good practice for file opening and
        # do other process for file handling.
        with open(filepath, newline='', mode='r') as csvfile:

            # Create reader.
            reader = csv.reader(csvfile)

            # Skip to the starting row.
            reader = itertools.islice(reader, start_row, None)
            
            # Check flag solution.
            if(flag_solutions_id == 0):
                for _ in range(chunk_size):
                    row = next(reader, None)
                    # Skip 'None' rows and 'Solution ID' colum.
                    if row:
                        list_rows.append(row[1:])
            else:
                for _ in range(chunk_size):
                    row = next(reader, None)
                    # Skip 'None' rows.
                    if row:
                        list_rows.append(row)
        
        # Output.
        return list_rows

    # Exception handling return section.
    except FileNotFoundError:
        raise ValueError(f"File in {filepath} does not exist.")
    except OSError:
        raise ValueError("Cannot read the file due to an I/O error.")
    except StopIteration:
        # Return the rows read so far if iteration ends prematurely.
        return list_rows 
    
def ReadInputCSV_threads(
        filepath: str, 
        n_workers: int = 1, 
        solutions_id_colum: int = 0
    ) -> Tuple[list, int, list[np.ndarray]]:
    """
    ReadInputCSV(function): This function reads a .csv matrix with the following formats:

    (ID_Solutions).
    Gene 1;Gene 2;Gene 3;...;Gene n
    Solution 1;1;2;4;1
    Solution 2;1;1;2;4;1
    Solution 3;1;1;2;4;1
    ...;
    Solution k;1;1;2;4;1

    (No ID_solutions)
    Gene 1;Gene 2;Gene 3;...;Gene n
    1;2;4;1
    1;1;2;4;1
    1;1;2;4;1
    ...;
    1;1;2;4;1

    Parameters:
    - filepath (str): Path that allocates the file in the computer.
    - n_jobs (int): Number of threads that are gonna read equals portions of the input file. 
    if you use -1 the function use all core of your computer.
    solutions_id_colum (int): flag that indicates presence of id for solutions.
        - 0: There is solutions ID.
        - 1: Opposite of 0.
        
    Returns:
    - genes (list): List of genes names (or ID's).
    - n (int): Number of genes.
    - Matrix (numpy.ndarray): Clusters results matrix.
    """
    # Exception handling: This allows the program to handle common
    # errors in this task.
    try:
        # Empty file verification.
        if os.path.getsize(filepath) == 0:
            raise ValueError("Empty file.")

        # Max core checking.
        max_cores = os.cpu_count()
        if max_cores is None:
            raise ValueError("Unable to determine the number of CPU cores.")

        # Valid n_jobs in input.
        if n_workers == -1:
            n_workers = max_cores 
        elif n_workers <= 0:
            raise ValueError("n_jobs must be a positive integer or -1.")
        
        # with: good practice for file opening and
        # do other process for file handling.
        with open(filepath, mode = "r") as file:
            # Start reading process.
            Csvfile = csv.reader(file)

            # List of genes and number of genes.
            genes = next(Csvfile)
            n = len(genes)

            # Calculates total of rows.
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                total_rows = mmapped_file.read().count(b'\n') - 1

        # Chunk size.
        chunk_size = (total_rows + n_workers - 1) // n_workers
        
        # Calling multiple threads to read the file. 'ProcessPoolExecutor'
        # handle all threads for the reading process.
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i in range(n_workers):
                start_row = i*chunk_size + 1
                futures.append(executor.submit(read_csv_part, filepath, start_row, chunk_size, solutions_id_colum))
        results = [future.result() for future in futures]
        Matrix = list(itertools.chain.from_iterable(results))
        
        # Reception.
        results = [future.result() for future in futures]
        # Merge all results.
        Matrix = list(itertools.chain.from_iterable(results))

        # Remove id solutions space from first row.
        if(solutions_id_colum == 0):
            genes.pop(0)
            n = n-1
        
        # Output.
        return genes, n, np.array(Matrix, dtype=int)
    
    # Exception handling return section.
    except FileNotFoundError:
        raise ValueError(f"File at {filepath} does not exist.")
    except OSError:
        raise ValueError("Cannot read the file due to an I/O error.")
    except ValueError as e:
        raise ValueError(f"Input error: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {e}")