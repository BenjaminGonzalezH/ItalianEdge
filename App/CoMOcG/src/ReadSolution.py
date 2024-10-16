######### Libraries #########
import itertools            # Eficient iterations.
import os                   # OS callings.
import numpy as np          # Math and Structures.
import csv                  # Read csv.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.

######### Functions #########

def read_csv_part(filepath, start_row, chunk_size):
    """
    read_csv_part(function)
        Input:
            - filepath: Path that allocates the file
            in the computer.
            - start_row: Row where the thread is going to
            start reading the file.
            - chunk_size: Amount of rows that thread is
            gonna read.
        Output: 
            - list_rows: All the rows readed.
        
        Description: This function is use inside of ReadInputCSV
        that creates threads and every one is read a portion of the
        file allocated in 'filepath'.
    """
    # Exception handling: This allows the program to handle common
    # errors in this task. Also, the threads that fails are not
    # going to hander the process.
    try:
        # Output variable.
        list_rows = []

        # with: good practice for file opening and
        # do other process for file handling.
        with open(filepath, newline='') as csvfile:

            # Reading process.
            reader = csv.reader(csvfile)

            # Skip rows.
            for _ in range(start_row):
                next(reader, None)
            for _ in range(chunk_size):
                row = next(reader, None)
                # Skip 'None' rows and 'Solition ID'. 
                if row:
                    list_rows.append(row[1:])
        
        # Output.
        return list_rows

    # Error Handling section.
    except FileNotFoundError:
        raise ValueError(f"File in {filepath} does not exist.")
    except OSError:
        raise ValueError("Cannot read the file due to an I/O error.")
    except StopIteration:
        return list_rows
    
def read_csv_part_NoID(filepath, start_row, chunk_size):
    """
    read_csv_part(function)
        Input:
            - filepath: Path that allocates the file
            in the computer.
            - start_row: Row where the thread is going to
            start reading the file.
            - chunk_size: Amount of rows that thread is
            gonna read.
        Output: 
            - list_rows: All the rows readed.
        
        Description: This function is use inside of ReadInputCSV_NoID
        that creates threads and every one is read a portion of the
        file allocated in 'filepath'.
    """    
    # Exception handling: This allows the program to handle common
    # errors in this task. Also, the threads that fails are not
    # going to hander the process.
    try:
        # Output variable.
        list_rows = []

        # with: good practice for file opening and
        # do other process for file handling.
        with open(filepath, newline='') as csvfile:
            
            # Reading process.
            reader = csv.reader(csvfile)
            
            # Skip rows
            for _ in range(start_row):
                next(reader, None)
            for _ in range(chunk_size):
                # Skip 'None' rows.
                row = next(reader, None)
                if row:
                    list_rows.append(row)
        
        # Output.
        return list_rows

    # Error Handling section.
    except FileNotFoundError:
        raise ValueError(f"File in {filepath} does not exist.")
    except OSError:
        raise ValueError("Cannot read the file due to an I/O error.")
    except StopIteration:
        return list_rows

def ReadInputCSV(filepath, n_threads=1):
    """
    ReadInputCSV(function)
    Input:
            - filepath: Path that allocates the file
            in the computer.
            - n_threads: number of threads that are gonna read
            equals portions of the input file.
        Output: 
            - genes: List of genes names. 
            - n: number of genes.
            - Matrix: Clusters results matrix.
        
        Description: This function reads a .csv matrix with the
        following format:

        Gene 1;Gene 2;Gene 3;...;Gene n
        Solution 1;1;2;4;1
        Solution 2;1;1;2;4;1
        Solution 3;1;1;2;4;1
        ...;
        Solution k;1;1;2;4;1

        Every cell allocates the cluster of their respective gene in the
        solution. Taking advantage of threads we can read rows faster 
        distributing the work using the cores of your CPU.
    """
    # Exception handling: This allows the program to handle common
    # errors in this task. Also, the threads that fails are not
    # going to hander the process.
    try:
        # Empty file verification.
        if os.path.getsize(filepath) == 0:
            raise ValueError("Empty file.")
        
        # with: good practice for file opening and
        # do other process for file handling.
        with open(filepath, mode = "r") as file:
            # Start reading process.
            Csvfile = csv.reader(file)

            # List of genes.
            genes = next(Csvfile)
            n = len(genes)

            # Calculates total of rows.
            total_rows = sum(1 for _ in file)

        # Determinate rows for threads.
        chunk_size = (total_rows + n_threads - 1) // n_threads
        
        # Calling multiple threads to read the file. 'ThreadPoolExecutor'
        # handle all threads for the reading process.
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            # Results of the threads.
            futures = []

            # Reading inicialitazion.
            for i in range(n_threads):
                start_row = i*chunk_size + 1
                futures.append(executor.submit(read_csv_part, filepath,start_row, chunk_size))
        # Reception.
        results = [future.result() for future in futures]
        # Merge all results.
        Matrix = list(itertools.chain.from_iterable(results))
        
        # Output.
        return genes, n, np.array(Matrix, dtype=int)
    
    # Error Handling section.
    except FileNotFoundError:
        raise ValueError(f"File in {filepath} does not exists.")
    except OSError:
        raise ValueError("Can not read the file due to a I/O error.")

def ReadInputCSV_NoID(filepath, n_threads=1):
    """
    ReadInputCSV(function)
    Input:
            - filepath: Path that allocates the file
            in the computer.
            - n_threads: number of threads that are gonna read
            equals portions of the input file.
        Output: 
            - genes: List of genes names. 
            - n: number of genes.
            - Matrix: Clusters results matrix.
        
        Description: This function reads a .csv matrix with the
        following format:

        Gene 1;Gene 2;Gene 3;...;Gene n
        1;2;4;1
        1;1;2;4;1
        1;1;2;4;1
        ...;
        1;1;2;4;1

        Every cell allocates the cluster of their respective gene in the
        solution. Taking advantage of threads we can read rows faster 
        distributing the work using the cores of your CPU.
    """
    # Exception handling: This allows the program to handle common
    # errors in this task. Also, the threads that fails are not
    # going to hander the process.
    try:
        # Empty file verification.
        if os.path.getsize(filepath) == 0:
            raise ValueError("Empty file.")
        
        # with: good practice for file opening and
        # do other process for file handling.
        with open(filepath, mode = "r") as file:
            # Start reading process.
            Csvfile = csv.reader(file)

            # List of genes.
            genes = next(Csvfile)
            n = len(genes)

            # Calculates total of rows.
            total_rows = sum(1 for _ in file)

        # Determinate rows for threads.
        chunk_size = (total_rows + n_threads - 1) // n_threads
        
        # Calling multiple threads to read the file. 'ThreadPoolExecutor'
        # handle all threads for the reading process.
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            # Results of the threads.
            futures = []
            
            # Reading inicialitazion.
            for i in range(n_threads):
                start_row = i*chunk_size + 1
                futures.append(executor.submit(read_csv_part_NoID, filepath,start_row, chunk_size))
        # Reception.
        results = [future.result() for future in futures]
        # Merge all results.
        Matrix = list(itertools.chain.from_iterable(results))
        
        # Output.
        return genes, n, np.array(Matrix, dtype=int)
    
    # Error Handling section.
    except FileNotFoundError:
        raise ValueError(f"File in {filepath} does not exists.")
    except OSError:
        raise ValueError("Can not read the file due to a I/O error.")