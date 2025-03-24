######### Libraries #########
import seaborn as sns                                       # Barplot.
import matplotlib.pyplot as plt                             # Graph construction.
import numpy as np                                          # Efficient Math Operations.
import rpy2.robjects as robjects                            # Transport Python data to R enviroment.
from rpy2.robjects import pandas2ri                         # R dataframe into Pandas dataframe.
import rpy2.robjects.vectors as r_vectors                   # Transport Python data to R enviroment (vectors).
import plotly.express as px                                 # HTML interactive plots.
import pandas as pd                                         # Dataframes.
import os                                                   # OS callings.

######### Own Libraries #########
from GoEnrischment import convert_symbols_to_entrez


def run_r_script(script_name, *args):
    """
    Run an R script stored in the 'R_Scripts' folder.

    Parameters:
    script_name (str): The name of the script file (without .R extension).
    args: Arguments to pass to the R script.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dir_path = os.path.join(base_dir, "R_Scripts")
    file_path = os.path.join(dir_path, f"{script_name}.R")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"R script not found: {file_path}")
    
    # Load and execute the script
    with open(file_path, "r", encoding="utf-8") as file:
        r_code = file.read()
        r_func = robjects.r(r_code)
        return r_func(*args)

######### Functions #########

"""
This block contains all main functions.
"""

def plot_gene_ratio(
        df: pd.DataFrame, 
        save_path:str = None, 
        show_flag:bool = True) -> None:
    """
    plot_gene_ratio (function): Plot the GeneRatio for GO terms and 
    optionally save the plot.

    Parameters:
    - df: DataFrame with enriched GO terms and associated data.
    - save_path: Path to save the plot.
    - show_flag: Flag for display plot.
    """
    try:
        # Size of figure.
        plt.figure(figsize=(20, 10))

        # Take dataframe necesary data.
        sorted_df = df.sort_values("GeneRatio", ascending=False)
        size = sorted_df["Count"]
        color = sorted_df["p.adjust"]
        values = sorted_df["GeneRatio"].apply(lambda x: round(eval(x), 2))

        # Draw plot.
        scatter = plt.scatter(
            x=values,
            y=sorted_df["Description"],
            s=size * 10,  # Scale size for better visualization.
            c=color,
            cmap="coolwarm",  # Color map ranging from blue to red.
            alpha=0.7,
            edgecolors="w",
            linewidth=0.5
        )

        # Labels.
        plt.xlabel("Gene Ratio")
        plt.ylabel("GO Terms")
        plt.title("Gene Ratio for GO Terms")
        plt.colorbar(scatter, label="Adjusted p-value")
        plt.tight_layout()

        # Save.
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved at: {save_path}")

        # Display.
        if show_flag:
            plt.show()
        else:
            plt.close()

    except Exception as e:
        print(f"Error plotting Gene Ratio: {e}")

def plot_gene_ratio_interactive(
        df: pd.DataFrame, 
        save_path:str = 'geneRatioPlot.html') -> None:
    """
    plot_gene_ratio_interactive (function): Same as 'plot_gene_ratio' function.
    Create a HTML that allocates the equivalent plot but interactive.

    Parameters:
    df (pd.DataFrame): DataFrame with enriched GO terms and associated data.
    save_path (str): Path to save the interactive plot as an HTML file.
    show_flag (bool): Whether to display the plot in the browser.
    """
    try:
        # Take dataframe necesary data.
        sorted_df = df.sort_values("GeneRatio", ascending=False)
        sorted_df['GeneRatio'] = sorted_df['GeneRatio'].apply(lambda x: round(eval(x), 2))
        
        # Draw plot.
        fig = px.scatter(
            sorted_df,
            x='GeneRatio',
            y='Description',
            size='Count',
            color='p.adjust',
            color_continuous_scale='viridis',
            title="Gene Ratio for GO Terms",
            labels={"GeneRatio": "Gene Ratio", "p.adjust": "Adjusted p-value"},
            hover_data={'Description': True, 'Count': True, 'p.adjust': True}
        )
        
        # Save to HTML.
        fig.write_html(save_path)
        print(f"Interactive Gene Ratio plot saved as: {save_path}")

    except Exception as e:
        print(f"Error creating interactive Gene Ratio plot: {str(e)}")

def plot_qscore(
        df: pd.DataFrame, 
        save_path:str = None, 
        show_flag:bool = True) -> None:
    """
    plot_qscore (function): Plot the qScore (negative log of qvalue) 
    for GO terms and optionally save the plot.

    Parameters:
    - df: DataFrame with enriched GO terms and associated data.
    - save_path: Path to save the plot.
    - show_flag: Flag for display plot.
    """
    try:
        # Take dataframe necesary data.
        df = df.copy()
        df['p.adjust'] = -np.log10(df['p.adjust'])
        sorted_df = df.sort_values("p.adjust", ascending=False)

        # Draw plot.
        plt.figure(figsize=(10, 6))
        sns.barplot(
            y=sorted_df["Description"],
            x=sorted_df["p.adjust"],
            hue=None,
            palette="coolwarm"
        )
        plt.xlabel("-log10(p.adjust)")
        plt.ylabel("GO Terms")
        plt.title("qScore for GO Terms")
        plt.tight_layout()

        # Draw.
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved at: {save_path}")

        # Display.
        if show_flag:
            plt.show()
        else:
            plt.close()

    except Exception as e:
        print(f"Error plotting qScore: {e}")

def plot_qscore_interactive(
        df: pd.DataFrame, 
        save_path:str = 'Qplot.html'):
    """
    plot_qscore_interactive (function): Same as 'plot_qscore' function.
    Create a HTML that allocates the equivalent plot but interactive.

    Parameters:
    df (pd.DataFrame): DataFrame with enriched GO terms and associated data.
    save_path (str): Path to save the interactive plot as an HTML file.
    """
    try:
        df['qScore'] = -np.log10(df['p.adjust'])  # Calculate qScore.

        # Create an interactive bar chart.
        fig = px.bar(
            df.sort_values('qScore', ascending=False),
            y='Description',
            x='qScore',
            title="qScore for GO Terms",
            labels={"qScore": "-log10(p.adjust)"},
            color='qScore',
            color_continuous_scale='viridis'
        )
        
        # Save to HTML.
        fig.write_html(save_path)
        print(f"Interactive qScore plot saved as: {save_path}")

    except Exception as e:
        print(f"Error creating interactive qScore plot: {str(e)}")

def plot_go_interaction_network_rpy2_1(gene_list, organism="org.Hs.eg.db", aspect="BP", 
                                     similarity_threshold=0.7, save_path=None, convert_ids=True,
                                     keytype = "SYMBOL", 
                                     width=1000, height=800, res=150):
    """
    Generate an interaction network for GO terms using R and rpy2, with adjustable image size and resolution.
    
    Parameters:
    - gene_list (list): List of gene symbols or Entrez IDs.
    - organism (str): Organism database to use in R (e.g., "org.Hs.eg.db").
    - aspect (str): GO aspect to focus on ('BP', 'MF', 'CC').
    - similarity_threshold (float): Minimum Wang similarity to create an edge.
    - save_path (str): Path to save the graph. If None, the graph is displayed in R.
    - convert_ids (bool): Whether to convert gene symbols to Entrez IDs.
    - width (int): Width of the image in pixels (default: 1000).
    - height (int): Height of the image in pixels (default: 800).
    - res (int): Resolution of the image in ppi (default: 150).

    Returns:
    - None
    """
    try:
        # Validate the input gene list
        if not isinstance(gene_list, list) or len(gene_list) == 0:
            raise ValueError("Input must be a non-empty list of genes.")
        
        # Convert gene symbols to Entrez IDs if needed
        if convert_ids:
            entrez_ids = convert_symbols_to_entrez(gene_list, organism, keytype)
            if len(entrez_ids) == 0:
                raise ValueError("No valid Entrez IDs could be derived from the input gene list.")
            print(f"Converted {len(gene_list)} gene symbols to {len(entrez_ids)} Entrez IDs")
        else:
            entrez_ids = gene_list
        
        # Convert the gene list to an R-compatible vector
        pandas2ri.activate()
        r_gene_list = robjects.StrVector(entrez_ids)

        # R code to generate GO interaction network with customizable image size and resolution
        r_code = f"""
        function(gene_list, save_path, similarity_threshold, organism, aspect, width, height, res) {{
            library(clusterProfiler)
            library(enrichplot)
            library({organism})
            
            tryCatch({{
                # Perform GO enrichment analysis
                ego <- enrichGO(gene = gene_list, ont = "{aspect}", OrgDb = {organism})
                edox <- pairwise_termsim(ego)
                
                if (!is.null(save_path)) {{
                    png(save_path, width = width, height = height, res = res)
                }}
                
                # Plot emapplot
                print(emapplot(edox))
                
                if (!is.null(save_path)) {{
                    dev.off()
                }}
            }}, error = function(e) {{
                stop("R encountered an error: ", e$message)
            }})
        }}
        """

        # Create the R function from the code
        r_func = robjects.r(r_code)

        # Call the R function with the converted gene list and image parameters
        r_func(r_gene_list, save_path, similarity_threshold, organism, aspect, width, height, res)
        
        print(f"GO interaction network created successfully{' and saved at ' + save_path if save_path else ''}")

    except Exception as e:
        print(f"Error generating GO interaction network: {str(e)}")

def plot_go_interaction_network_rpy2(gene_list, save_path, similarity_threshold=0.7, organism="org.Hs.eg.db", 
                        aspect="BP", width=1000, height=800, res=150, convert_ids=True, keytype="SYMBOL"):
    """
    Wrapper function to call the R script for generating a GO interaction network.
    
    Parameters:
    - gene_list (list): List of gene symbols or Entrez IDs.
    - save_path (str): Path to save the plot.
    - similarity_threshold (float): Minimum Wang similarity to create an edge.
    - organism (str): Organism database to use in R.
    - aspect (str): GO aspect ('BP', 'MF', 'CC').
    - width (int): Width of the image.
    - height (int): Height of the image.
    - res (int): Resolution of the image.

    Returns:
    - None
    """
    try:
        # Validate the input gene list
        if not isinstance(gene_list, list) or len(gene_list) == 0:
            raise ValueError("Input must be a non-empty list of genes.")
        
        # Convert gene symbols to Entrez IDs if needed
        if convert_ids:
            entrez_ids = convert_symbols_to_entrez(gene_list, organism, keytype)
            if len(entrez_ids) == 0:
                raise ValueError("No valid Entrez IDs could be derived from the input gene list.")
            print(f"Converted {len(gene_list)} gene symbols to {len(entrez_ids)} Entrez IDs")
        else:
            entrez_ids = gene_list
        
        result = run_r_script("GO_intertactive_network", entrez_ids, save_path, similarity_threshold, organism, aspect, width, height, res)
        print(f"GO Network successfully created and saved at {save_path}")
        return result
    except Exception as e:
        print(f"Error generating GO network: {str(e)}")

def create_go_tree_rpy2(df, aspect='BP', max_nodes=50, save_path=None):
    """
    Generates a GO DAG with GO IDs and term names, colored by significance.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with GO analysis results.
        Must contain columns: 'ID', 'Description', 'p.adjust', 'Count'
    aspect : str
        GO aspect ('BP', 'MF', 'CC').
    max_nodes : int
        Maximum number of nodes to display in the graph.
    save_path : str, optional
        Path to save the graph. If None, displays on screen.
    """
    try:
        if df.empty or not all(col in df.columns for col in ['ID', 'Description', 'p.adjust']):
            raise ValueError("DataFrame is missing required columns ('ID', 'Description', 'p.adjust').")
        
        # Convert GO IDs and metadata to R vectors
        go_ids = r_vectors.StrVector(df['ID'].tolist())
        
        # Convert dictionaries properly
        p_adjust_dict = robjects.ListVector({str(k): v for k, v in zip(df['ID'], df['p.adjust'])})
        desc_dict = robjects.ListVector({str(k): str(v) for k, v in zip(df['ID'], df['Description'])})  # Convertir a str
        
        # Map aspect to correct ontology
        aspect_map = {"BP": "GOBPPARENTS", "MF": "GOMFPARENTS", "CC": "GOCCPARENTS"}
        go_parents = aspect_map.get(aspect, "GOBPPARENTS")
        
        # Modified R code with term names in nodes
        r_code = f"""
        function(go_ids, aspect, max_nodes, save_path, p_adjust_values, descriptions) {{
            library(GO.db)
            library(GOstats)
            library(graph)
            library(Rgraphviz)
            
            # Get terms and create graph
            go_terms <- unique(go_ids)
            gograph <- GOGraph(go_terms, {go_parents})
            
            # Filter nodes
            valid_nodes <- intersect(nodes(gograph), go_terms)
            gograph <- subGraph(valid_nodes, gograph)
            
            # Limit nodes if needed
            if (numNodes(gograph) > max_nodes) {{
                valid_nodes <- sample(valid_nodes, max_nodes)
                gograph <- subGraph(valid_nodes, gograph)
            }}
            
            # Function to get significance level (1-9)
            get_significance_level <- function(p_value) {{
                if (is.na(p_value)) return(0)
                if (p_value <= 5e-10) return(9)
                if (p_value <= 5e-9) return(8)
                if (p_value <= 5e-8) return(7)
                if (p_value <= 5e-7) return(6)
                if (p_value <= 5e-6) return(5)
                if (p_value <= 5e-5) return(4)
                if (p_value <= 5e-4) return(3)
                if (p_value <= 5e-3) return(2)
                if (p_value <= 0.05) return(1)
                return(0)
            }}
            
            # Function to get color based on significance level
            get_node_color <- function(p_value) {{
                level <- get_significance_level(p_value)
                colors <- c("#FFFFFF", "#FFF7EC", "#FEE8C8", "#FDD49E", 
                           "#FDBB84", "#FC8D59", "#EF6548", "#D7301F",
                           "#B30000", "#7F0000")
                return(colors[level + 1])
            }}
            
            # Create node labels with GO ID and term name
            node_labels <- sapply(nodes(gograph), function(x) {{
                node_labels <- sapply(nodes(gograph), function(x) {{
                p_val <- p_adjust_values[[x]]
                desc <- descriptions[[x]]
                sig_level <- get_significance_level(p_val)
                
                # Manejar NA en descripciones
                if (is.null(desc) || is.na(desc)) {{
                    desc <- "Unknown Term"
                }}
                
                # Forzar salto de línea si el texto es demasiado largo
                desc_wrapped <- paste(strwrap(desc, width=30), collapse="\\n")
                
                sprintf("%s\\n(p=%0.2e)", desc_wrapped, p_val)
                }})
            }})
            
            # Set node attributes
            nAttrs <- list()
            nAttrs$label <- node_labels
            names(nAttrs$label) <- nodes(gograph)
            
            # Set colors based on significance
            node_colors <- sapply(nodes(gograph), 
                                function(x) get_node_color(p_adjust_values[[x]]))
            names(node_colors) <- nodes(gograph)
            nAttrs$fillcolor <- node_colors
            
            # Set other attributes
            nAttrs$shape <- rep("box", length(nodes(gograph)))
            names(nAttrs$shape) <- nodes(gograph)
            nAttrs$fontsize <- rep(20, length(nodes(gograph)))
            names(nAttrs$fontsize) <- nodes(gograph)
            
            # Edge attributes
            eAttrs <- list()
            eAttrs$arrowhead <- rep("vee", length(edges(gograph)))
            names(eAttrs$arrowhead) <- edges(gograph)
            
            # Generate plot
            if (!is.null(save_path)) {{
                pdf(save_path, width=15, height=12)
            }}
            
            # Create layout with top to bottom direction
            lay <- layoutGraph(gograph, layoutType="dot", 
                             attrs=list(graph=list(rankdir="TB")))
            
            # Plot graph
            plot(lay,
                 main=paste("GO", aspect, "DAG - Heat Response Terms"),
                 nodeAttrs=nAttrs,
                 edgeAttrs=eAttrs,
                 attrs=list(
                     node=list(
                         shape="box",
                         style="filled",
                         width=3.5,  # Increased width for better text display
                         height=1.2
                     ),
                     edge=list(
                         color="black",
                         dir="forward"
                     )
                 ))
            
            # Add legend for significance levels
            legend("bottomright", 
                   legend=paste("p≤", c("0.05", "5e-3", "5e-4", "5e-5",
                                      "5e-6", "5e-7", "5e-8", "5e-9", "5e-10")),
                   fill=c("#FFF7EC", "#FEE8C8", "#FDD49E", "#FDBB84",
                          "#FC8D59", "#EF6548", "#D7301F", "#B30000",
                          "#7F0000"),
                   border="black",
                   title="Significance Levels")
            
            if (!is.null(save_path)) {{
                dev.off()
            }}
            
            return(valid_nodes)
        }}
        """
        
        # Create and execute R function
        r_func = robjects.r(r_code)
        
        # Convert dictionaries to R lists
        p_adjust_r = robjects.ListVector(p_adjust_dict)
        desc_r = robjects.ListVector(desc_dict)
        
        used_nodes = r_func(go_ids, aspect, max_nodes, save_path, p_adjust_r, desc_r)
        
        print(f"GO DAG created successfully{' and saved to ' + save_path if save_path else ''}")
        print(f"Number of GO terms used: {len(used_nodes)}")
        
    except Exception as e:
        raise Exception(f"Error creating GO DAG: {str(e)}")