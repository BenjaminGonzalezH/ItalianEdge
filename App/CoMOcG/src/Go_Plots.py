import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as robjects
import seaborn as sns
from rpy2.robjects import pandas2ri
from GoEnrischment import convert_symbols_to_entrez
from rpy2.robjects.packages import importr

def map_genes_to_go_terms(df):
    """
    Relaciona genes con los términos GO en los que están involucrados.
    
    Parámetros:
    df (pd.DataFrame): DataFrame con dos columnas: 'GO_term' y 'Gene_IDs', donde 'Gene_IDs'
                        contiene los IDs de los genes asociados a cada término GO.

    Devuelve:
    dict: Diccionario donde las claves son los IDs de los genes y los valores son listas de términos GO
          asociados a cada gen.
    """
    gene_to_go = {}

    # Iterar sobre cada fila del DataFrame
    for _, row in df.iterrows():
        go_term = row['ID']
        gene_ids = row['geneID']
        gene_ids = [str(x) for x in gene_ids.split('/')]
        
        # Iterar sobre los genes asociados a cada término GO
        for gene in gene_ids:
            if gene not in gene_to_go:
                gene_to_go[gene] = []
            gene_to_go[gene].append(go_term)
    
    return gene_to_go

def plot_gene_ratio(df, save_path=None, show_flag = True):
    """
    Plot the GeneRatio for GO terms and optionally save the plot.

    Parameters:
    df (pd.DataFrame): DataFrame with enriched GO terms and associated data.
    save_path (str, optional): Path to save the plot. If None, the plot is shown.

    Returns:
    None: Displays or saves the plot.
    """
    try:
        plt.figure(figsize=(20, 10))
        sorted_df = df.sort_values("GeneRatio", ascending=False)
        size = sorted_df["Count"]  # Assuming 'Count' column represents the number of genes
        color = sorted_df["p.adjust"]  # Assuming 'p.adjust' column represents the adjusted p-values
        values = sorted_df["GeneRatio"].apply(lambda x: round(eval(x), 2))

        scatter = plt.scatter(
            x=values,
            y=sorted_df["Description"],
            s=size * 10,  # Scale size for better visualization
            c=color,
            cmap="coolwarm",  # Color map ranging from blue to red
            alpha=0.7,
            edgecolors="w",
            linewidth=0.5
        )

        plt.xlabel("Gene Ratio")
        plt.ylabel("GO Terms")
        plt.title("Gene Ratio for GO Terms")
        plt.colorbar(scatter, label="Adjusted p-value")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved at: {save_path}")

        if show_flag:
            plt.show()
        else:
            plt.close()

    except Exception as e:
        print(f"Error plotting Gene Ratio: {e}")

def plot_qscore(df, save_path=None, show_flag = True):
    """
    Plot the qScore (negative log of qvalue) for GO terms and optionally save the plot.

    Parameters:
    df (pd.DataFrame): DataFrame with enriched GO terms and associated data.
    save_path (str, optional): Path to save the plot. If None, the plot is shown.

    Returns:
    None: Displays or saves the plot.
    """
    try:
        df = df.copy()  # Avoid modifying original DataFrame
        df['p.adjust'] = -np.log10(df['p.adjust'])
        sorted_df = df.sort_values("p.adjust", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            y=sorted_df["Description"],
            x=sorted_df["p.adjust"],
            palette="coolwarm"
        )
        plt.xlabel("-log10(p.adjust)")
        plt.ylabel("GO Terms")
        plt.title("qScore for GO Terms")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved at: {save_path}")

        if show_flag:
            plt.show()
        else:
            plt.close()

    except Exception as e:
        print(f"Error plotting qScore: {e}")

def plot_go_interaction_network_rpy2(gene_list, organism="org.Hs.eg.db", aspect="BP", 
                                     similarity_threshold=0.7, save_path=None, convert_ids=True):
    """
    Generate an interaction network for GO terms using R and rpy2.
    
    Parameters:
    - gene_list (list): List of gene symbols or Entrez IDs.
    - organism (str): Organism database to use in R (e.g., "org.Hs.eg.db").
    - aspect (str): GO aspect to focus on ('BP', 'MF', 'CC').
    - similarity_threshold (float): Minimum Wang similarity to create an edge.
    - save_path (str): Path to save the graph. If None, the graph is displayed in R.
    - convert_ids (bool): Whether to convert gene symbols to Entrez IDs.

    Returns:
    - None
    """
    try:
        # Validate the input gene list
        if not isinstance(gene_list, list) or len(gene_list) == 0:
            raise ValueError("Input must be a non-empty list of genes.")
        
        # Convert gene symbols to Entrez IDs if needed
        if convert_ids:
            entrez_ids = convert_symbols_to_entrez(gene_list, organism)
            if len(entrez_ids) == 0:
                raise ValueError("No valid Entrez IDs could be derived from the input gene list.")
            print(f"Converted {len(gene_list)} gene symbols to {len(entrez_ids)} Entrez IDs")
        else:
            entrez_ids = gene_list
        
        # Convert the gene list to an R-compatible vector
        pandas2ri.activate()
        r_gene_list = robjects.StrVector(entrez_ids)

        # R code to generate GO interaction network
        r_code = f"""
        function(gene_list, save_path, similarity_threshold, organism, aspect) {{
            library(clusterProfiler)
            library(enrichplot)
            library({organism})
            
            tryCatch({{
                # Perform GO enrichment analysis
                ego <- enrichGO(gene = gene_list, ont = "{aspect}", OrgDb = {organism})
                edox <- pairwise_termsim(ego)
                
                if (!is.null(save_path)) {{
                    png(save_path)
                }}
                
                # Plot emapplot and treeplot
                print(emapplot(edox))

                if (!is.null(save_path)) {{
                    dev.off()
                }}
            }}, error = function(e) {{
                stop("R encountered an error: ", e$message)
            }})
        }}
        """

        # Create R function from the R code
        r_func = robjects.r(r_code)

        # Call the R function with the converted gene list
        r_func(r_gene_list, save_path, similarity_threshold, organism, aspect)
        
        print(f"GO interaction network created successfully{' and saved at ' + save_path if save_path else ''}")

    except Exception as e:
        print(f"Error generating GO interaction network: {str(e)}")

def create_go_tree_rpy2(df, aspect='BP', max_nodes=50, save_path=None):
    """
    Utiliza rpy2 para generar un gráfico GOgraph con datos de Python.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con resultados de análisis GO.
        Debe contener columnas: 'ID', 'Description', 'p.adjust', 'Count'
    aspect : str
        GO aspect ('BP', 'MF', 'CC').
    max_nodes : int
        Número máximo de nodos a mostrar en el gráfico.
    save_path : str, optional
        Ruta para guardar el gráfico. Si es None, se muestra en pantalla.
    """
    try:
        # Configurar entorno R
        #base, utils, go_db, gostats, graph, rgraphviz = setup_r_environment()
        
        # Convertir el DataFrame a formato R
        go_ids = robjects.vectors.StrVector(df['ID'].tolist())
        
        # Crear el código R para la visualización
        r_code = """
        function(go_ids, aspect, max_nodes, save_path) {
            library(GO.db)
            library(GOstats)
            library(graph)
            library(Rgraphviz)
            
            # Crear el grafo GO
            go_terms <- unique(go_ids)
            gograph <- GOGraph(go_terms, GOBPPARENTS)
            
            # Limitar el número de nodos
            if (numNodes(gograph) > max_nodes) {
                gograph <- subGraph(sample(nodes(gograph), max_nodes), gograph)
            }
            
            # Generar el gráfico
            if (!is.null(save_path)) {
                pdf(save_path)
            }
            
            plot(gograph, main = paste("GO", aspect, "Graph"))
            
            if (!is.null(save_path)) {
                dev.off()
            }
        }
        """
        
        # Crear la función R y ejecutarla
        r_func = robjects.r(r_code)
        r_func(go_ids, aspect, max_nodes, save_path)
        
        print(f"Gráfico GO creado exitosamente{' y guardado en ' + save_path if save_path else ''}")
        
    except Exception as e:
        raise Exception(f"Error creando el gráfico GO: {str(e)}")