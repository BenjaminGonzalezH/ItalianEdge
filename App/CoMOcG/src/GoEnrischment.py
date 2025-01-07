# Import
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas as pd

def convert_symbols_to_entrez(gene_symbols, organism="org.Hs.eg.db"):
    """
    convert_symbols_to_entrez(function): 
    Convert gene symbols to Entrez IDs.

    Parameters:
    - gene_symbols (list): List of gene symbols.
    - organism (str): Organism database to use.

    Returns:
    - list: List of corresponding Entrez IDs.
    """
    # Load required R packages.
    robjects.r(f'''
        library({organism})
        library(clusterProfiler)
    ''')
    
    # Convert Python list to R vector and assign it.
    r_genes = robjects.StrVector(gene_symbols)
    robjects.r.assign("gene_symbols", r_genes)
    
    # R code for conversion.
    r_code = (
        'entrez_ids <- mapIds(x = ' + organism + ', '
        'keys = gene_symbols, '
        'column = "ENTREZID", '
        'keytype = "SYMBOL", '
        'multiVals = "first"); '
        'entrez_ids <- na.omit(entrez_ids)'
    )
    
    # Run the conversion.
    robjects.r(r_code)
    
    # Get the converted IDs from R.
    entrez_ids = list(robjects.r('entrez_ids'))
    
    return entrez_ids

def setup_r_environment():
    """
    setup_r_environment(function): 
    Configure the R environment and install necessary packages.

    Returns:
    - dict: References to the imported R packages.
    """
    try:
        pandas2ri.activate()
        
        # Import base R packages
        base = importr('base')
        utils = importr('utils')
        
        # List of required R packages
        required_packages = [
            'clusterProfiler', 'org.Hs.eg.db', 'DOSE',
            'GO.db', 'GOstats', 'graph', 'Rgraphviz'
        ]
        
        def is_package_installed(package_name):
            """
            Check if an R package is installed.
            """
            return base.requireNamespace(package_name, quietly=True)[0]
        
        def install_package(package_name):
            """
            Install an R package if not already installed.
            """
            if not is_package_installed(package_name):
                print(f"Installing R package: {package_name}")
                utils.install_packages(package_name)
            else:
                print(f"R package already installed: {package_name}")
        
        # Install and load all required packages
        for package in required_packages:
            install_package(package)
        
        # Import installed packages
        loaded_packages = {}
        for package in required_packages:
            try:
                loaded_packages[package] = importr(package)
                print(f"Successfully loaded R package: {package}")
            except Exception as e:
                print(f"Failed to load R package {package}: {e}")
        
        return loaded_packages
    
    except Exception as e:
        raise Exception(f"Error configuring R environment: {str(e)}")

def perform_go_enrichment(gene_list, organism="org.Hs.eg.db", ont="BP", convert_ids=True):
    """
    perform_go_enrichment(function): 
    Perform GO enrichment analysis on a list of genes.

    Parameters:
    - gene_list (list): List of gene symbols or Entrez IDs.
    - organism (str): Organism database to use.
    - ont (str): GO ontology to use (BP: Biological Process, MF: Molecular Function, CC: Cellular Component).
    - convert_ids (bool): Whether to convert gene symbols to Entrez IDs.

    Returns:
    - pandas.DataFrame: Enrichment results.
    """
    try:
        # Convert gene symbols to Entrez IDs if needed.
        if convert_ids:
            entrez_ids = convert_symbols_to_entrez(gene_list, organism)
            print(f"Converted {len(gene_list)} gene symbols to {len(entrez_ids)} Entrez IDs")
        else:
            entrez_ids = gene_list
        
        if not entrez_ids:
            raise ValueError("No valid Entrez IDs were provided.")
        
        # Convert Python list to R vector.
        r_genes = robjects.StrVector(entrez_ids)
        robjects.r.assign("gene_list", r_genes)
        
        # Load necessary libraries in R.
        robjects.r(f'''
            library(clusterProfiler)
            library({organism})
        ''')
        
        # Perform enrichment analysis.
        robjects.r(f'''
            go_result <- enrichGO(
                gene = gene_list,
                OrgDb = {organism},
                ont = "{ont}",
                keyType = 'ENTREZID',
                readable = TRUE
            )
        ''')
        
        # Extract the results.
        r_results = robjects.r("as.data.frame(go_result)")
        
        # Convert R DataFrame to pandas DataFrame.
        result_df = pandas2ri.rpy2py(r_results)
        
        return result_df
    except Exception as e:
        print(f"An error occurred during GO enrichment analysis: {e}")
        return pd.DataFrame()

def calculate_wang_distance_matrix(gene_list, organism="org.Hs.eg.db", ont="BP", convert_ids=True):
    """
    Calculate a matrix of Wang semantic distances for a list of genes.
    """
    try:
        # Validate the input gene list
        if len(gene_list) < 2:
            raise ValueError("The gene list must contain at least two entries to calculate distances.")
        
        # Convert gene symbols to Entrez IDs if needed
        if convert_ids:
            entrez_ids = convert_symbols_to_entrez(gene_list, organism)
            print(f"Converted {len(gene_list)} gene symbols to {len(entrez_ids)} Entrez IDs")
        else:
            entrez_ids = gene_list
        
        if not entrez_ids:
            raise ValueError("No valid Entrez IDs were provided.")

        # Convert the gene list to an R vector
        r_gene_list = robjects.StrVector(entrez_ids)
        robjects.r.assign("gene_list", r_gene_list)

        # Load necessary R libraries and create a GO database object
        robjects.r(f'''
            library(GOSemSim)
            go_db <- godata(annoDb = "{organism}", 
                            ont = "{ont}", 
                            computeIC = TRUE)
        ''')

        # Calculate the similarity matrix in R
        sim_matrix_df = robjects.r('''
            sim_matrix <- mgeneSim(genes = gene_list, 
                                   semData = go_db, 
                                   measure = "Wang")
            sim_matrix[is.na(sim_matrix)] <- 0  # Replace NA with 0
            as.data.frame(sim_matrix)  # Convert to DataFrame
        ''')
        
        # Convert R DataFrame to pandas DataFrame
        sim_matrix_df = pandas2ri.rpy2py(sim_matrix_df)

        # Match dimensions: Use only genes present in the matrix
        included_genes = list(sim_matrix_df.columns)  # Extract genes in the matrix
        sim_matrix_df = sim_matrix_df.loc[included_genes, included_genes]

        # Update DataFrame index and column names
        sim_matrix_df.index = included_genes
        sim_matrix_df.columns = included_genes

        print(f"Final matrix dimensions: {sim_matrix_df.shape}")
        return sim_matrix_df

    except Exception as e:
        print(f"Error calculating Wang distance matrix: {e}")
        return pd.DataFrame()
