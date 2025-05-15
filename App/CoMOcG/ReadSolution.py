######### Libraries #########
import itertools                                    # Eficient iterations.
import os                                           # OS callings.
import numpy as np                                  # Efficient Math Operations.
import csv                                          # Read csv.
from concurrent.futures import ThreadPoolExecutor   # Thread Administration.
import mmap                                         # Mapping data in memory.
from typing import Tuple                            # Multiple returns doc.
import pandas as pd                                 # Dataframe managment.

# Note: the return is not a Tuple, this is used for
# allocates all the multiple elements of output that
# would have one function.

######### Functions #########

"""
This block contains all main functions.
"""

def ReadSolutionsFile(
        filepath: str, 
        format: str = "csv") -> Tuple[np.ndarray, list[str]]:
    """
    ReadSolutionsFile (function): Read clustering solutions, represented by a collection
    of integer arrays, from a file. The primate format of this files has to be compatible 
    with pandas supported files.

    Parameters:
        - filepath: Location of the file in your PC (Machine). Must contain filename and
          extention.
        - format: File extention of the input file.
    """
    # Setting string into lowercase to avoid input errors in format.
    case_string = format.lower()
    
    # Configured list of formats.
    format_list = ["csv", "fwf (fixed-width file)", "pkl (Python Pickle Format)"]

    # Checking format.
    try:
        match case_string:
            
            ######################################################### CSV.
            case "csv":
                # Deducing separator from csv.
                with open(filepath,"r",encoding="utf-8") as f:
                    sample = f.read(1024)
                    delimeter = csv.Sniffer().sniff(sample).delimiter

                # Reading file.
                dataframe = pd.read_csv(filepath, sep=delimeter)
                df_cleaned = dataframe.loc[:, ~dataframe.columns.str.contains("Unnamed")]

                # Extracting solutions and genes.
                Matrix = df_cleaned.to_numpy()
                Genes = list(df_cleaned.columns)

            ######################################################### Fixed-Width Text File.
            case "fwf" | "fixed-width file":
                # Reading file.
                dataframe = pd.read_fwf(filepath)
                df_cleaned = dataframe.loc[:, ~dataframe.columns.str.contains("Unnamed")]

                # Extracting solutions and genes.
                Matrix = dataframe.to_numpy()
                Genes = list(dataframe.columns)

            ######################################################### Python Pickle Format.
            case "pkl" | "python pickle format":
                # Reading file.
                dataframe = pd.read_pickle(filepath)
                df_cleaned = dataframe.loc[:, ~dataframe.columns.str.contains("Unnamed")]

                # Check if pkl is a dataframe.
                if isinstance(dataframe, pd.DataFrame):
                    # Extracting solutions and genes.
                    Matrix = dataframe.to_numpy()
                    Genes = list(dataframe.columns)
                else:
                    print("pkl file is not a dataframe")
                    # Error return.
                    Matrix = np.zeros((2,2))
                    Genes = ["",""]
        
            ######################################################### Python Pickle Format.
            case _:
                # Feedback message.
                print(f"Not Supported format: {format} - Better Try: {format_list}.")

    except FileNotFoundError:
        raise FileNotFoundError(f"File at {filepath} not found.")
    except KeyboardInterrupt:
        raise RuntimeError(f"User shutdown process by Keyboard command.")
    except Exception as e:
        raise RuntimeError(f"Something went wrong.\nDetails: {e}")
    else:
        print(f"File in: {filepath} succesfully readed.\nSolutions and Genes obtained.")
        return Matrix, Genes