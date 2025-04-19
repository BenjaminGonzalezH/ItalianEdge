######### Libraries #########
import numpy as np                                  # Efficient Math Operations.
import plotly.graph_objects as go                   # Interactive plots.

######### Functions #########

"""
This block contains all main functions.
"""

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
    plot_dual_heatmap_two_colors(function): Generate a heatmap with different color 
    scales for upper and lower triangular matrices. Includes highlighting of both the 
    hovered cell and its symmetric counterpart.
    
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
                hovertext[i][j] = f"Solution 2: {j}<br>Solution 1: {i}<br>Wang: {matrix_lower[i][j]:.2f}"
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
    
    # Add two traces for highlighted cells (will be updated via JavaScript)
    # One for the hovered cell
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            symbol='square',
            size=20,
            color='rgba(255, 0, 0, 0.0)',
            line=dict(color='red', width=2)
        ),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # One for the symmetric cell
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            symbol='square',
            size=20,
            color='rgba(255, 0, 0, 0.0)',
            line=dict(color='red', width=2)
        ),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(constrain="domain"),
        annotations=[
            dict(
                text="Pase el cursor sobre una celda para ver su coordenada simétrica",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.15
            )
        ],
        hovermode='closest'
    )
    
    # Save the basic plot
    fig.write_html(save_filepath, include_plotlyjs='cdn', full_html=True)
    
    try:
        # Add custom JavaScript for symmetric coordinate highlighting
        with open(save_filepath, 'r') as file:
            html_content = file.read()
        
        # Find where the Plotly div is defined
        import re
        div_pattern = r'<div id="([^"]+)"'
        match = re.search(div_pattern, html_content)
        
        if match:
            div_id = match.group(1)
            
            # JavaScript to highlight symmetric coordinates on hover
            js_code = f"""
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    var myPlot = document.getElementById('{div_id}');
                    
                    myPlot.on('plotly_hover', function(data) {{
                        if (!data.points || data.points.length === 0) return;
                        
                        var point = data.points[0];
                        var xCoord = point.x;
                        var yCoord = point.y;
                        
                        // Update highlight traces
                        var updateCurrentCell = {{
                            x: [[xCoord]],
                            y: [[yCoord]]
                        }};
                        
                        var updateSymmetricCell = {{
                            x: [[yCoord]],
                            y: [[xCoord]]
                        }};
                        
                        // Update the highlight traces (index 2 and 3)
                        Plotly.restyle(myPlot, updateCurrentCell, [2]);
                        Plotly.restyle(myPlot, updateSymmetricCell, [3]);
                    }});
                    
                    myPlot.on('plotly_unhover', function() {{
                        // Clear highlights
                        var clearHighlights = {{
                            x: [[null]],
                            y: [[null]]
                        }};
                        
                        Plotly.restyle(myPlot, clearHighlights, [2, 3]);
                    }});
                }});
            </script>
            """
            
            html_content = html_content.replace('</body>', js_code + '</body>')
            
            # Save the modified HTML
            with open(save_filepath, 'w') as file:
                file.write(html_content)
                
            print(f"✅ Heatmap combinado con dos colores guardado en: {save_filepath}")
            print("🖱️ Se ha añadido resaltado de coordenadas simétricas al pasar el cursor")
                
    except Exception as e:
        print(f"✅ Heatmap combinado guardado en: {save_filepath}")
        print(f"ℹ️ Nota: usando versión básica del heatmap (sin resaltado de coordenadas simétricas)")
        print(f"Error al modificar HTML: {str(e)}")