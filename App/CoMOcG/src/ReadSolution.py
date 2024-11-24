######### Libraries #########
import itertools                                    # Eficient iterations.
import os                                           # OS callings.
import numpy as np                                  # Math and Structures.
import csv                                          # Read csv.
from concurrent.futures import ProcessPoolExecutor  # Process Administration.
import mmap                                         # Mapping data in memory.

######### Functions #########

def read_csv_part(filepath, start_row, chunk_size, flag_solutions_id = 0):
    """
    read_csv_part(function)
        Input:
            - filepath: Path that allocates the file
            in the computer.
            - start_row: Row where the process is going to
            start reading the file.
            - chunk_size: Amount of rows that process is
            gonna read.
            - flag_id: Flag that indicates if the CSV
            file uses ID for indicate order in his solutions.
                - 0: There is solutions ID.
                - 1: Opposite of 0.
        Output: 
            - list_rows: All the rows readed.
        
        Description: This function is use inside of ReadInputCSV for
        multiple process that creates for input reading.
    """
    # Exception handling: This allows the program to handle common
    # errors in this task.
    try:
        # Validate inputs
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

    # Error Handling section.
    except FileNotFoundError:
        raise ValueError(f"File in {filepath} does not exist.")
    except OSError:
        raise ValueError("Cannot read the file due to an I/O error.")
    except StopIteration:
        # Return the rows read so far if iteration ends prematurely.
        return list_rows 


def ReadInputCSV(filepath, n_jobs = 1, solutions_id_colum = 0):
    """
    ReadInputCSV(function)
    Input:
            - filepath: Path that allocates the file
            in the computer.
            - n_jobs: number of process (core) that are gonna read
            equals portions of the input file. if you use -1 the function
            use all core of your computer.
        Output:
            - genes: List of genes names. 
            - n: number of genes.
            - Matrix: Clusters results matrix.
        
        Description: This function reads a .csv matrix with the
        following formats:

        Gene 1;Gene 2;Gene 3;...;Gene n
        Solution 1;1;2;4;1
        Solution 2;1;1;2;4;1
        Solution 3;1;1;2;4;1
        ...;
        Solution k;1;1;2;4;1

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
        if n_jobs == -1:
            n_jobs = max_cores 
        elif n_jobs <= 0:
            raise ValueError("n_jobs must be a positive integer or -1.")
        elif n_jobs > max_cores:
            raise ValueError(f"n_jobs ({n_jobs}) cannot exceed the number of available CPU cores ({max_cores}).")
        
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
        chunk_size = (total_rows + n_jobs - 1) // n_jobs
        
        # Calling multiple threads to read the file. 'ProcessPoolExecutor'
        # handle all threads for the reading process.
        # Usar ProcessPoolExecutor para paralelizar la lectura
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []

            for i in range(n_jobs):
                start_row = i * chunk_size + 1  # Saltar encabezado
                futures.append(
                    executor.submit(
                        read_csv_part, filepath, start_row, chunk_size, solutions_id_colum
                    )
                )
        
        # Reception.
        results = [future.result() for future in futures]
        # Merge all results.
        Matrix = list(itertools.chain.from_iterable(results))
        
        # Output.
        return genes, n, np.array(Matrix, dtype=int)
    
    # Error Handling section.
    except FileNotFoundError:
        raise ValueError(f"File at {filepath} does not exist.")
    except OSError:
        raise ValueError("Cannot read the file due to an I/O error.")
    except ValueError as e:
        raise ValueError(f"Input error: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {e}")