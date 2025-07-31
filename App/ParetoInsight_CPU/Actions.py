######### Libraries #########
import os                                           # OS callings.
import numpy as np                                  # Efficient Math Operations.
import pandas as pd                                 # Dataframe managment.
import matplotlib                                   # Plots.

# Configurations: 
matplotlib.use('Agg')                               # Conf: No use of GUI interface -> conflict when I use Threads.

######### Functions #########

"""
This block contains all main functions.
"""

def save_matrix(
        matrix: np.ndarray, 
        save_filepath: str) -> None:
    """
    save_matrix (function): Save a matrix in binary format for faster I/O.
    
    Parameters:
    - matrix: Matrix to save.
    - save_filepath: Path to save the matrix (must include name and extension). Also,
      you need to ensure to write the path of the file in your computer.

    Returns:
    - None in data, just create a file with the matrix.
    """
    try:
        # Create directories if needed.
        directory = os.path.dirname(save_filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save as a binary .npy file for faster I/O. This is a alternative
        # to manage large amount of data.
        np.save(save_filepath, matrix)
        print(f"Matrix saved in binary format at: {save_filepath}.npy")

    except Exception as e:
        print(f"Unexpected error: {e}")

def save_matrix_uncompresed(
        matrix: np.ndarray, 
        save_filepath: str) -> None:
    """
    save_matrix(function): Save a matrix that is a result from a function.

    Parameters:
    - matrix: Matrix.
    - proportion_filepath: Path to save the matrix (needs to have the name and
      extension).

    Returns:
    - None in data, just create a compresed file version of the matrix.
    """
    try:
        # Create directories if needed.
        directory = os.path.dirname(save_filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save matrix.
        np.savetxt(save_filepath, matrix, delimiter=",", fmt="%.6f")
        print(f"Proportion matrix saved at: {save_filepath}")

    except Exception as e:
        print(f"Unexpected error: {e}")

def load_and_display_matrix(
        filepath: str
        ) -> np.ndarray:
    """
    load_and_display_matrix (function): Load and display a matrix stored in a .npy file.
    
    Parameters:
    - filepath: Path to the .npy file (without the .npy extension).
    
    Returns:
    - matrix: Loaded matrix.
    """
    try:
        # Check file existence.
        if not os.path.exists(filepath):
            print(f"Error: File {filepath}.npy not found")
            return None

        # load matrix from .npy file.
        matrix = np.load(filepath)
        
        # Display.
        print("loaded Matrix:")
        print(matrix)

        # Matrix return.
        return matrix

    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

def save_dataframe(
        dataframe: pd.DataFrame, 
        filepath: str, 
        format: str ="csv"
        ) -> None:
    """
    save_dataframe (function): Save a DataFrame with faster formats like Parquet.
    
    Parameters:
    - dataframe (pd.DataFrame): DataFrame to save.
    - filepath (str): Path to save the file.
    - format (str): "csv", "excel", or "parquet". Default is "csv".

    Return:
    - None, create a file with info allocated in dataframe.
    """
    try:
        if format.lower() == "csv":
            dataframe.to_csv(filepath, index=False)
            print(f"Results saved as CSV at: {filepath}")
        elif format.lower() == "excel":
            dataframe.to_excel(filepath, index=False, engine='openpyxl')
            print(f"Results saved as Excel file at: {filepath}")
        elif format.lower() == "parquet":
            dataframe.to_parquet(filepath, index=False)
            print(f"Results saved as Parquet file at: {filepath}")
        else:
            raise ValueError("Unsupported format. Use 'csv', 'excel', or 'parquet'.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
