######### Libraries #########
import os                                           # OS callings.
import numpy as np                                  # Efficient Math Operations.
import matplotlib.pyplot as plt                     # Graph construction.
import plotly.graph_objects as go                   # Interactive plots.
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

def load_and_display_matrix(
        filepath: str
        ) -> np.ndarray:
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

def plot_html_heatmap(
        matrix: np.ndarray, 
        save_filepath: str = 'heatmap.html', 
        x_label: str = '', 
        y_label: str = '', 
        title: str = 'Heatmap', 
        color: str = 'Viridis',
        z_label: str = 'Valor',
        tooltip_format: str = "X: %{x}<br>Y: %{y}<br>Valor: %{z:.2f}"):  
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
        z_label (str): Label for the color bar (z-axis). Default is 'Valor'.
        tooltip_format (str): Custom tooltip format using Plotly syntax.
    """
    try:
        # Validate the input matrix.
        if not isinstance(matrix, np.ndarray) or len(matrix.shape) != 2:
            raise ValueError("The 'matrix' input must be a 2D NumPy array.")

        # Create the heatmap figure.
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            colorscale=color,
            colorbar=dict(title=z_label),
            hovertemplate=tooltip_format  # Personalized Toolip.
        ))

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            autosize=True,
            xaxis=dict(scaleanchor="y", constrain="domain"),
            yaxis=dict(constrain="domain")
        )

        # Save the figure as an HTML file.
        fig.write_html(save_filepath)
        print(f"Interactive heatmap saved as an HTML file at: {save_filepath}")

    except ValueError as ve:
        print(f"Validation Error: {ve}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

def plot_dual_heatmap_single_hover(
    matrix_upper: np.ndarray,
    matrix_lower: np.ndarray,
    save_filepath: str = 'dual_heatmap.html',
    x_label: str = 'Solutions',
    y_label: str = 'Solutions',
    title: str = 'Jaccard vs Wang',
    colorscale: str = 'Viridis',
):
    if matrix_upper.shape != matrix_lower.shape:
        raise ValueError("Las matrices deben tener el mismo tamaño.")

    n = matrix_upper.shape[0]
    z = np.full((n, n), np.nan)
    hovertext = []

    for i in range(n):
        row = []
        for j in range(n):
            if i < j:
                z[i][j] = matrix_upper[i][j]
                text = f"Solution 1: {j}<br>Solution 2: {i}<br>Jaccard: {matrix_upper[i][j]:.2f}"
            elif i > j:
                z[i][j] = matrix_lower[i][j]
                text = f"Solution 1: {j}<br>Solution 2: {i}<br>Wang: {matrix_lower[i][j]:.2f}"
            else:
                text = ""
            row.append(text)
        hovertext.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=hovertext,
        hoverinfo='text',
        colorscale=colorscale,
        colorbar=dict(title='Jaccard / Wang')
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(constrain="domain")
    )

    fig.write_html(save_filepath)
    print(f"✅ Heatmap combinado guardado en: {save_filepath}")

def plot_dual_heatmap_two_colors(
    matrix_upper: np.ndarray,
    matrix_lower: np.ndarray,
    save_filepath: str = 'dual_heatmap.html',
    x_label: str = 'Solutions',
    y_label: str = 'Solutions',
    title: str = 'Jaccard vs Wang',
    colorscale_upper: str = 'Viridis',
    colorscale_lower: str = 'Plasma',
):
    """
    Generate a heatmap with different color scales for upper and lower triangular matrices.
    
    Args:
        matrix_upper: Upper triangular matrix data (Jaccard)
        matrix_lower: Lower triangular matrix data (Wang)
        save_filepath: Path to save the HTML plot
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
        colorscale_upper: Color scale for upper triangular matrix
        colorscale_lower: Color scale for lower triangular matrix
    """
    if matrix_upper.shape != matrix_lower.shape:
        raise ValueError("Las matrices deben tener el mismo tamaño.")
    
    n = matrix_upper.shape[0]
    
    # Create separate data for upper and lower triangular matrices
    z_upper = np.full((n, n), np.nan)
    z_lower = np.full((n, n), np.nan)
    
    # Combined hover text matrix
    hovertext = [['' for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i < j:  # Upper triangle (Jaccard)
                z_upper[i][j] = matrix_upper[i][j]
                hovertext[i][j] = f"Solution 1: {j}<br>Solution 2: {i}<br>Jaccard: {matrix_upper[i][j]:.2f}"
            elif i > j:  # Lower triangle (Wang)
                z_lower[i][j] = matrix_lower[i][j]
                hovertext[i][j] = f"Solution 1: {j}<br>Solution 2: {i}<br>Wang: {matrix_lower[i][j]:.2f}"
            else:  # Diagonal
                hovertext[i][j] = f"Solution: {i}"
    
    # Create figure with two heatmap traces
    fig = go.Figure()
    
    # Upper triangular heatmap (Jaccard)
    fig.add_trace(go.Heatmap(
        z=z_upper,
        customdata=np.array(hovertext),
        hovertemplate="%{customdata}<extra></extra>",
        colorscale=colorscale_upper,
        showscale=True,
        colorbar=dict(
            title='Jaccard',
            x=1.0,
            y=0.75,
            len=0.4,
        )
    ))
    
    # Lower triangular heatmap (Wang)
    fig.add_trace(go.Heatmap(
        z=z_lower,
        customdata=np.array(hovertext),
        hovertemplate="%{customdata}<extra></extra>",
        colorscale=colorscale_lower,
        showscale=True,
        colorbar=dict(
            title='Wang',
            x=1.0,
            y=0.25,
            len=0.4,
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(constrain="domain")
    )
    
    # Save the plot
    fig.write_html(save_filepath)
    print(f"✅ Heatmap combinado con dos colores guardado en: {save_filepath}")

def save_dataframe(
        dataframe: pd.DataFrame, 
        filepath: str, 
        format: str ="csv"
        ):
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

def Pairs_Ordered(
        matrix: np.ndarray, 
        desc: bool = False,
        diagonal: bool = False
    ) -> pd.DataFrame:
    """
    Order pairs (upper triangular) of a matrix considerating
    values that allocates.
    
    Parameters:
    - matrix: Input matrix.
    - desc: Order (ascending and descending). False for ascending.
    
    Returns:
    - pd.DataFrame with columns ['solution_ID_1', 'solution_ID_2', 'value'] ordered.
    """
    try:
        if not isinstance(matrix, np.ndarray):
            raise TypeError("The input 'matrix' must be a NumPy ndarray.")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("The input matrix must be a square 2D array.")
        
        # Obtain pairs of the upper triangular (no diagonal).
        if(diagonal):
            indices_valores = [((i, j), matrix[i, j]) 
                            for i in range(matrix.shape[0]) 
                            for j in range(i+1, matrix.shape[1])]
        else:
            indices_valores = [((i, j), matrix[i, j]) 
                            for i in range(matrix.shape[0]) 
                            for j in range(i, matrix.shape[1])]
        
        # Order by value.
        indices_valores.sort(key=lambda x: x[1], reverse=desc)
        
        # Create rows for dataframe.
        id1 = [i for ((i, j), val) in indices_valores]
        id2 = [j for ((i, j), val) in indices_valores]
        values = [val for ((i, j), val) in indices_valores]
        
        # Construct dataframe.
        df = pd.DataFrame({
            'solution_ID_1': id1,
            'solution_ID_2': id2,
            'value': values
        })
        
        return df

    except Exception as e:
        print(f"Error in Pairs_Ordered: {e}")
        return pd.DataFrame(columns=['solution_ID_1', 'solution_ID_2', 'value'])