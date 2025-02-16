######### Libraries #########
import os                                           # OS callings.
import numpy as np                                  # Efficient Math Operations.
import matplotlib.pyplot as plt                     # Graph construction.
import pandas as pd
import plotly.graph_objects as go

######### Functions #########
def save_matrix(matrix: np.ndarray, save_filepath: str) -> None:
    """
    Optimized save_matrix: Save a matrix in binary format for faster I/O.
    
    Parameters:
    - matrix (np.ndarray): Matrix to save.
    - save_filepath (str): Path to save the matrix (must include name and extension).
    """
    try:
        # Create directories if needed
        directory = os.path.dirname(save_filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save as a binary .npy file for faster I/O
        np.save(save_filepath, matrix)
        print(f"Matrix saved in binary format at: {save_filepath}.npy")

    except Exception as e:
        print(f"Unexpected error: {e}")

def save_matrix_uncompresed(matrix: np.ndarray, 
                save_filepath: str) -> None:
    """
    save_matrix(function): Save a matrix that is a result from a function.

    Parameters:
    - matrix (np.ndarray): Matrix.
    - proportion_filepath (str): Path to save the matrix (needs to have the name).
    """
    try:
        # Create directories if needed
        directory = os.path.dirname(save_filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save proportion matrix
        np.savetxt(save_filepath, matrix, delimiter=",", fmt="%.6f")
        print(f"Proportion matrix saved at: {save_filepath}")

    except Exception as e:
        print(f"Unexpected error: {e}")

def plot_heatmap_matrix(matrix: np.ndarray, save_filepath: str = None,
                        x_label: str = '',
                        y_label: str = '',
                        title: str = 'Heatmap',
                        color: str = 'viridis', show_flag: bool = True):
    """
    Plots a heatmap from a NumPy array and optionally saves it to a file.

    Args:
        matrix (np.ndarray): The 2D array to visualize as a heatmap.
        save_filepath (str): Path to save the heatmap image (optional).
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the heatmap. Default is 'Heatmap'.
        color (str): Colormap for the heatmap. Default is 'viridis'.
        show_flag (bool): Whether to display the heatmap. Default is True.
    """
    try:
        # Validar la matriz de entrada
        if not isinstance(matrix, np.ndarray) or len(matrix.shape) != 2:
            raise ValueError("La entrada 'matrix' debe ser una matriz NumPy bidimensional.")

        # Validar el colormap
        if color not in plt.colormaps():
            raise ValueError(f"'{color}' no es un colormap válido. Use uno de: {plt.colormaps()}")

        # Configurar la figura
        plt.figure(figsize=(20, 10))
        plt.imshow(matrix, cmap=color, interpolation='nearest')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.colorbar()
        plt.title(title)

        # Guardar la figura si se proporciona una ruta válida
        if save_filepath:
            plt.savefig(save_filepath, format=save_filepath.split('.')[-1])
            print(f"Heatmap guardado en: {save_filepath}")

        # Mostrar la figura si está habilitado
        if show_flag:
            plt.show()

        # Cerrar la figura para liberar memoria
        plt.close()

    except ValueError as ve:
        print(f"Error de validación: {ve}")
    except FileNotFoundError:
        print(f"Error: No se pudo guardar el archivo en '{save_filepath}'. Verifique la ruta.")
    except Exception as e:
        print(f"Error inesperado: {e}")

def plot_html_heatmap(matrix: np.ndarray, 
                      save_filepath: str = 'heatmap.html', 
                      x_label: str = '', 
                      y_label: str = '', 
                      title: str = 'Heatmap', 
                      color: str = 'Viridis'):
    """
    Creates an interactive heatmap and saves it as an HTML file that can be opened in a browser.

    Args:
        matrix (np.ndarray): The 2D array to visualize as a heatmap.
        save_filepath (str): Path to save the heatmap as an HTML file. Default is 'heatmap.html'.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the heatmap. Default is 'Heatmap'.
        color (str): Colormap for the heatmap. Default is 'Viridis'.
    """
    try:
        # Validate the input matrix
        if not isinstance(matrix, np.ndarray) or len(matrix.shape) != 2:
            raise ValueError("The 'matrix' input must be a 2D NumPy array.")

        # Create the heatmap figure
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
    Optimized save_dataframe: Save a DataFrame with faster formats like Parquet.
    
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