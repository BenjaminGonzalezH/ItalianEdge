######### Libraries #########
import os                                           # OS callings.
import numpy as np                                  # Efficient Math Operations.
import matplotlib.pyplot as plt                     # Graph construction.
import plotly.graph_objects as go                   # Interactive plots.

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
    - matrix (np.ndarray): Matrix to save.
    - save_filepath (str): Path to save the matrix (must include name and extension). Also,
      you need to ensure to write the path of the file in your computer.
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

def load_and_display_matrix(filepath: str) -> np.ndarray:
    """
    load_and_display_matrix (function): Load and display a matrix stored in a .npy file.
    
    Parameters:
    - filepath (str): Path to the .npy file (without the .npy extension).
    
    Returns:
    - np.ndarray: Loaded matrix.
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

def save_matrix_uncompresed(
        matrix: np.ndarray, 
        save_filepath: str) -> None:
    """
    save_matrix(function): Save a matrix that is a result from a function.

    Parameters:
    - matrix (np.ndarray): Matrix.
    - proportion_filepath (str): Path to save the matrix (needs to have the name and
      extension).
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

def plot_heatmap_matrix(
        matrix: np.ndarray, save_filepath: str = None,
        x_label: str = '',
        y_label: str = '',
        title: str = 'Heatmap',
        color: str = 'viridis', 
        show_flag: bool = True) -> None:
    """
    plot_heatmap_matrix (function): Plots a heatmap from a NumPy array 
    and optionally saves it to a file.

    Parameters:
        matrix (np.ndarray): The Matrix with data to plot.
        save_filepath (str): Path to save the heatmap image (optional).
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the heatmap. Default is 'Heatmap'.
        color (str): Colormap for the heatmap. Default is 'viridis'.
        show_flag (bool): Display instanly the heatmap. Default is True.
    """
    try:
        # Validate input matrix.
        if not isinstance(matrix, np.ndarray) or len(matrix.shape) != 2:
            raise ValueError("The input 'matrix' must be a 2D numpy array.")

        # Validate colormap option.
        if color not in plt.colormaps():
            raise ValueError(f"'{color}' is not a valid colormap. Use one of these options: {plt.colormaps()}")

        # Figure configuratrion.
        plt.figure(figsize=(20, 10))
        plt.imshow(matrix, cmap=color, interpolation='nearest')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.colorbar()
        plt.title(title)

        # Save image if have a valid path.
        if save_filepath:
            plt.savefig(save_filepath, format=save_filepath.split('.')[-1])
            print(f"Heatmap guardado en: {save_filepath}")

        # Display path if it is alowed.
        if show_flag:
            plt.show()

        # Close draw interface.
        plt.close()

    except ValueError as ve:
        print(f"Validation Error: {ve}")
    except FileNotFoundError:
        print(f"Error: Can not save '{save_filepath}'. Check file path.")
    except Exception as e:
        print(f"Unexpected Error: {e}")

def plot_html_heatmap(
        matrix: np.ndarray, 
        save_filepath: str = 'heatmap.html', 
        x_label: str = '', 
        y_label: str = '', 
        title: str = 'Heatmap', 
        color: str = 'Viridis'):
    """
    plot_html_heatmap (function): Creates an interactive heatmap and saves it 
    as an HTML file that can be opened in a browser.

    Parameters:
        matrix (np.ndarray): The Matrix with data to plot.
        save_filepath (str): Path to save the heatmap image (optional).
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the heatmap. Default is 'Heatmap'.
        color (str): Colormap for the heatmap. Default is 'viridis'.
    """
    try:
        # Validate the input matrix.
        if not isinstance(matrix, np.ndarray) or len(matrix.shape) != 2:
            raise ValueError("The 'matrix' input must be a 2D NumPy array.")

        # Create the heatmap figure.
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            colorscale=color,
            colorbar=dict(title="Value"),
        ))

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            autosize=True,
        )

        # Save the figure as an HTML file
        fig.write_html(save_filepath)
        print(f"Interactive heatmap saved as an HTML file at: {save_filepath}")

    except ValueError as ve:
        print(f"Validation Error: {ve}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

def save_dataframe(dataframe, filepath, format="csv"):
    """
    save_dataframe (function): Save a DataFrame with faster formats like Parquet.
    
    Parameters:
    - dataframe (pd.DataFrame): DataFrame to save.
    - filepath (str): Path to save the file.
    - format (str): "csv", "excel", or "parquet". Default is "csv".
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