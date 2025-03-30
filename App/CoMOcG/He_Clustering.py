######### Libraries #########
import traceback
import numpy as np                                                  # Efficient Math Operations.
import matplotlib.pyplot as plt                                     # Graph construction.
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster   # Create clustering.
from scipy.spatial.distance import squareform                       # Create dendogram.
import os                                                           # OS callings.
import matplotlib                                                   # Plots.
import plotly.graph_objects as go                                   # Interactive insert of graphs.

# Configurations: 
matplotlib.use('Agg')                               # Conf: No use of GUI interface -> conflict when I use Threads.

######### Functions #########

"""
This block contains all main functions.
"""
   
def He_clustering(
        distance_matrix: np.ndarray, 
        genes: list[str], 
        num_groups: int = 4,
        save_path: str = "dendrogram", 
        dendrogram_file: str = "dendrogram.html",
        method: str = "single") -> list:
    """
    He_clustering_interactive(function): Perform hierarchical clustering and generate an interactive dendrogram.

    Parameters:
    - distance_matrix (np.ndarray): Square distance matrix between genes.
    - genes (list[str]): Gene identifiers.
    - num_groups (int): Number of clusters for the consensus solution.
    - save_path (str): Directory to save the dendrogram.
    - dendrogram_file (str): Name of the output HTML file.
    - method (str): Linkage method to use. Default is "single".
    - show_flag (bool): Whether to display the dendrogram in the browser. Default is True.

    Returns:
    - list: Consensus clustering solution.
    """
    try:
        # Validate that the matrix is square.
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("Distance matrix must be square.")
        
        # Make sure the number of genes matches the dimension of the distance matrix.
        if len(genes) != distance_matrix.shape[0]:
            raise ValueError(f"Number of genes ({len(genes)}) does not match distance matrix dimensions ({distance_matrix.shape[0]})")
        
        # Check non zero groups.
        if num_groups < 1:
            raise ValueError("num_groups must be greater than 0.")
        
        # Convert to condensed form.
        condensed_dist_matrix = squareform(distance_matrix)

        # Perform hierarchical clustering.
        Z = linkage(condensed_dist_matrix, method=method)

        # Define consensus clusters.
        consensus_solution = fcluster(Z, num_groups, criterion='maxclust')

        # Generate labels with gene names and positions.
        labels = [f"{i}-{gene}" for i, gene in enumerate(genes)]

        # Generate the dendrogram data without plotting.
        dendrogram_data = dendrogram(Z, labels=labels, no_plot=True)
        
        # Create interactive dendrogram with plotly.
        fig = go.Figure()
        
        # Add the dendrogram traces.
        for i, d in enumerate(dendrogram_data['dcoord']):
            x = dendrogram_data['icoord'][i]
            y = d
            
            # Create a list of point pairs for each line segment.
            xs = []
            ys = []
            for j in range(len(x)-1):
                xs.extend([x[j], x[j+1], None])
                ys.extend([y[j], y[j+1], None])
            
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode='lines',
                line=dict(color='black'),
                hoverinfo='none'
            ))
        
        # Add labels.
        leaf_idx = dendrogram_data['leaves']
        leaf_labels = [labels[i] for i in leaf_idx]
        
        # Get the x-coordinates for leaf labels.
        # In the scipy dendrogram, leaf nodes are at positions 5, 15, 25, etc.
        leaf_positions = []
        icoord = dendrogram_data['icoord']
        for i in range(len(leaf_idx)):
            # Find the x-coordinate for this leaf.
            leaf_positions.append(5 + 10 * i)
        
        # Add cutoff line for clusters.
        cutoff_height = Z[-(num_groups - 1), 2]
        x_range = [0, 10 * len(leaf_idx)]
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=[cutoff_height, cutoff_height],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=f'{num_groups} Clusters'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Dendrogram with {num_groups} clusters',
            xaxis=dict(
                title='Genes (Index-Name)',
                tickmode='array',
                tickvals=leaf_positions,
                ticktext=leaf_labels,
                tickangle=45
            ),
            yaxis=dict(title='Distance'),
            height=800,
            width=1200,
            showlegend=False
        )
        
        # Create directory if it doesn't exist.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Save the figure.
        html_path = os.path.join(save_path, dendrogram_file)
        fig.write_html(html_path)
        print(f"Interactive dendrogram saved at: {html_path}")
        
        return consensus_solution
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()  # This will print the full traceback for debugging.
        return None