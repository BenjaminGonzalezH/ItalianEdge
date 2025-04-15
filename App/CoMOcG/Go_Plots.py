######### Libraries #########
import numpy as np                                          # Efficient Math Operations.
import rpy2.robjects as robjects                            # Transport Python data to R enviroment.
from rpy2.robjects import pandas2ri                         # R dataframe into Pandas dataframe.
import rpy2.robjects.vectors as r_vectors                   # Transport Python data to R enviroment (vectors).
import plotly.express as px                                 # HTML interactive plots.
import pandas as pd                                         # Dataframes.

######### Own Libraries #########
from CoMOcG.GoEnrischment import convert_symbols_to_entrez

######### Functions #########

"""
This block contains all main functions.
"""

def plot_gene_ratio(
        df: pd.DataFrame, 
        save_path:str = 'geneRatioPlot.html'
        ) -> None:
    """
    plot_gene_ratio (function): Create a HTML that allocates the gene ratio plot.

    Parameters:
    - df (pd.DataFrame): DataFrame with enriched GO terms and associated data.
    - save_path (str): Path to save the interactive plot as an HTML file.
    """
    try:
        # Take dataframe necesary data.
        sorted_df = df.sort_values(['p.adjust', 'GeneRatio', 'Count'], ascending=[True, False, False])
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
        save_path:str = 'Qplot.html'
        ) -> None:
    """
    plot_qscore (function): Same as 'plot_qscore' function.
    Create a HTML that allocates the equivalent plot but interactive.

    Parameters:
    df (pd.DataFrame): DataFrame with enriched GO terms and associated data.
    save_path (str): Path to save the interactive plot as an HTML file.
    """
    try:
        df['qScore'] = -np.log10(df['p.adjust'])  # Calculate qScore.

        # Create an interactive bar chart.
        fig = px.bar(
            df.sort_values('qScore', ascending=True),
            y='Description',
            x='qScore',
            title="qScore for GO Terms",
            labels={"qScore": "Qscore"},
            color='qScore',
            color_continuous_scale='viridis'
        )
        
        # Save to HTML.
        fig.write_html(save_path)
        print(f"Interactive qScore plot saved as: {save_path}")

    except Exception as e:
        print(f"Error creating interactive qScore plot: {str(e)}")

def plot_go_interaction_network_rpy2(
        gene_list: list[str], 
        organism: str = "org.Hs.eg.db", 
        aspect: str = "BP", 
        similarity_threshold: float = 0.7, 
        save_path: str= None, 
        convert_ids: bool= True,
        keytype: str = "SYMBOL", 
        width: int = 1000, 
        height: int = 800, 
        res: int = 150):
    """
    plot_go_interaction_network_rpy2 (function): Generate an interaction network for GO terms using R and rpy2, 
    with adjustable image size and resolution.
    
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

def create_go_tree_rpy2(
        df: pd.DataFrame, 
        aspect: str = 'BP', 
        max_nodes: int = 50, 
        save_path: str = None):
    """
    create_go_tree_rpy2 (function): Generates a GO DAG with GO IDs and term names, colored by significance.
    
    Parameters:
    - df: DataFrame with GO analysis results. Must contain columns: 'ID', 'Description', 'p.adjust', 'Count'
    aspect: GO aspect ('BP', 'MF', 'CC').
    - max_nodes: Maximum number of nodes to display in the graph.
    - save_path: Path to save the graph. If None, displays on screen.
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
                dynamic_width <- min(15, 0.25 * length(nodes(gograph)))
                dynamic_height <- min(12, 0.2 * length(nodes(gograph)))

                pdf(save_path, width=dynamic_width, height=dynamic_height)
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