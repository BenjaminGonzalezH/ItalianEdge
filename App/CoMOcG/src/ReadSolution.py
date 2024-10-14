######### Libraries #########
import itertools
import os
import numpy as np
import csv
from concurrent.futures import ThreadPoolExecutor

######### Functions #########

def read_csv_part(filepath, start_row, chunk_size):  
    try:
        list_rows = []
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            
            for _ in range(start_row):
                next(reader, None)

            for _ in range(chunk_size):
                row = next(reader, None)
                if row:
                    list_rows.append(row[1:])
        return list_rows

    except FileNotFoundError:
        raise ValueError(f"File in {filepath} does not exist.")
    except OSError:
        raise ValueError("Cannot read the file due to an I/O error.")
    except StopIteration:
        return list_rows
    
def read_csv_part_NoID(filepath, start_row, chunk_size):  
    try:
        list_rows = []
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            
            for _ in range(start_row):
                next(reader, None)

            for _ in range(chunk_size):
                row = next(reader, None)
                if row:
                    list_rows.append(row)
        return list_rows

    except FileNotFoundError:
        raise ValueError(f"File in {filepath} does not exist.")
    except OSError:
        raise ValueError("Cannot read the file due to an I/O error.")
    except StopIteration:
        return list_rows

def ReadInputCSV(filepath, chunk_size=10, n_threads=1):
    try:
        # Empty file verification.
        if os.path.getsize(filepath) == 0:
            raise ValueError("Empty file.")
        
        # Read First Line.
        # Open CSV file.
        with open(filepath, mode = "r") as file:
            # Start reading process.
            Csvfile = csv.reader(file)

            # List of genes.
            genes = next(Csvfile)
            n = len(genes)
        
        # Calling multiple threads to read the file.
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            for i in range(n_threads):
                start_row = i*chunk_size + 1
                futures.append(executor.submit(read_csv_part, filepath,start_row, chunk_size))
        results = [future.result() for future in futures]
        Matrix = list(itertools.chain.from_iterable(results))
        
        return genes, n, np.array(Matrix, dtype=int)
    
    except FileNotFoundError:
        raise ValueError(f"File in {filepath} does not exists.")
    except OSError:
        raise ValueError("Can not read the file due to a I/O error.")

def ReadInputCSV_NoID(filepath, chunk_size=10, n_threads=1):
    try:
        # Empty file verification.
        if os.path.getsize(filepath) == 0:
            raise ValueError("Empty file.")
        
        # Read First Line.
        # Open CSV file.
        with open(filepath, mode = "r") as file:
            # Start reading process.
            Csvfile = csv.reader(file)

            # List of genes.
            genes = next(Csvfile)
            n = len(genes)
        
        # Calling multiple threads to read the file.
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            for i in range(n_threads):
                start_row = i*chunk_size + 1
                futures.append(executor.submit(read_csv_part_NoID, filepath,start_row, chunk_size))
        results = [future.result() for future in futures]
        Matrix = list(itertools.chain.from_iterable(results))
        
        return genes, n, np.array(Matrix, dtype=int)
    
    except FileNotFoundError:
        raise ValueError(f"File in {filepath} does not exists.")
    except OSError:
        raise ValueError("Can not read the file due to a I/O error.")