######### Libraries #########
import plotly.express as px                                 # HTML interactive plots.
import pandas as pd                                         # Dataframes.


######### Functions #########

"""
This block contains all main functions.
"""

def plot_gene_ratio(
        df: pd.DataFrame, 
        save_path:str = 'gene_ratioPlot.html'
        ) -> None:
    """
    plot_gene_ratio (function): Create a HTML that allocates the gene ratio plot.

    Parameters:
    - df (pd.DataFrame): DataFrame with enriched GO terms and associated data.
    - save_path (str): Path to save the interactive plot as an HTML file.
    """
    try:
        # Take dataframe necesary data.
        sorted_df = df.sort_values(['p_value', 'gene_ratio', 'intersection_size'], ascending=[True, False, False])
        sorted_df['gene_ratio'] = sorted_df['gene_ratio']
        
        # Draw plot.
        fig = px.scatter(
            sorted_df,
            x='gene_ratio',
            y='name',
            size='intersection_size',
            color='p_value',
            color_continuous_scale='viridis',
            title="Gene Ratio for GO Terms",
            labels={"gene_ratio": "Gene Ratio", "p_value": "Adjusted p-value"},
            hover_data={'name': True, 'intersection_size': True, 'p_value': True}
        )
        
        # Save to HTML.
        fig.write_html(save_path)
        print(f"Interactive Gene Ratio plot saved as: {save_path}")

    except Exception as e:
        print(f"Error creating interactive Gene Ratio plot: {str(e)}")

def plot_qscore(
        df: pd.DataFrame, 
        save_path:str = 'Qplot.html'
        ) -> None:
    """
    plot_qscore (function): Same as 'plot_qscore' function.
    Create a HTML that allocates the equivalent plot but interactive.

    Parameters:
    df: DataFrame with enriched GO terms and associated data.
    save_path: Path to save the interactive plot as an HTML file.
    """
    try:
        df = df[df['gene_ratio'] > 0]
        # Create an interactive bar chart.
        fig = px.bar(
            df.sort_values('qscore', ascending=True),
            y='name',
            x='qscore',
            title="qscore for GO Terms",
            labels={"qscore": "Qscore"},
            color='qscore',
            color_continuous_scale='viridis'
        )
        
        # Save to HTML.
        fig.write_html(save_path)
        print(f"Interactive qscore plot saved as: {save_path}")

    except Exception as e:
        print(f"Error creating interactive qscore plot: {str(e)}")