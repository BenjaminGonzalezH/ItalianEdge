# Import
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np

def convert_symbols_to_entrez(gene_symbols, organism="org.Hs.eg.db"):
    """
    Convert gene symbols to Entrez IDs
    
    Parameters:
    gene_symbols (list): List of gene symbols
    organism (str): Organism database to use
    
    Returns:
    list: List of corresponding Entrez IDs
    """
    # Load required R packages
    robjects.r(f'''
        library({organism})
        library(clusterProfiler)
    ''')
    
    # Convert Python list to R vector and assign it
    r_genes = robjects.StrVector(gene_symbols)
    robjects.r.assign("gene_symbols", r_genes)
    
    # R code for conversion
    r_code = (
        'entrez_ids <- mapIds(x = ' + organism + ', '
        'keys = gene_symbols, '
        'column = "ENTREZID", '
        'keytype = "SYMBOL", '
        'multiVals = "first"); '
        'entrez_ids <- na.omit(entrez_ids)'
    )
    
    # Run the conversion
    robjects.r(r_code)
    
    # Get the converted IDs from R
    entrez_ids = list(robjects.r('entrez_ids'))
    
    return entrez_ids


def setup_r_environment():
    """Install and load required R packages"""
    pandas2ri.activate()
    # Create R vector of packages to install
    utils = importr('utils')
    
    # Define the packages we need
    packages = ['clusterProfiler', 'org.Hs.eg.db', 'DOSE']
    
    # Install packages if not already installed
    r_code = '''
        install_if_missing <- function(p) {
            if (!requireNamespace(p, quietly = TRUE)) {
                BiocManager::install(p)
            }
        }
    '''
    robjects.r(r_code)
    
    # Install BiocManager if not present
    robjects.r('if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")')
    
    # Install each package
    for pkg in packages:
        robjects.r(f'install_if_missing("{pkg}")')

def perform_go_enrichment(gene_list, organism="org.Hs.eg.db", ont="BP", convert_ids=True):
    """
    Perform GO enrichment analysis on a list of genes
    
    Parameters:
    gene_list (list): List of gene symbols or Entrez IDs
    organism (str): Organism database to use
    ont (str): GO ontology to use (BP: Biological Process, MF: Molecular Function, CC: Cellular Component)
    convert_ids (bool): Whether to convert gene symbols to Entrez IDs
    
    Returns:
    pandas.DataFrame: Enrichment results
    """
    try:
        # Convert gene symbols to Entrez IDs if needed
        if convert_ids:
            entrez_ids = convert_symbols_to_entrez(gene_list, organism)
            print(f"Converted {len(gene_list)} gene symbols to {len(entrez_ids)} Entrez IDs")
        else:
            entrez_ids = gene_list
        
        if not entrez_ids:
            raise ValueError("No valid Entrez IDs were provided.")
        
        # Convert Python list to R vector
        r_genes = robjects.StrVector(entrez_ids)
        robjects.r.assign("gene_list", r_genes)
        
        # Load necessary libraries in R
        robjects.r(f'''
            library(clusterProfiler)
            library({organism})
        ''')
        
        # Perform enrichment analysis
        robjects.r(f'''
            go_result <- enrichGO(
                gene = gene_list,
                OrgDb = {organism},
                ont = "{ont}",
                pAdjustMethod = "BH",
                pvalueCutoff = 0.05,
                qvalueCutoff = 0.2
            )
        ''')
        
        # Extract the results
        r_results = robjects.r("as.data.frame(go_result)")
        
        # Convert R DataFrame to pandas DataFrame
        result_df = pandas2ri.rpy2py(r_results)
        
        return result_df
    except Exception as e:
        print(f"An error occurred during GO enrichment analysis: {e}")
        return pd.DataFrame()
    
def perform_wang_similarity(dataframe, organism="org.Hs.eg.db", ont="BP"):
    """
    Calculate Wang semantic similarity between GO terms in a DataFrame.
    
    Parameters:
    dataframe (pd.DataFrame): DataFrame with a column 'ID' containing GO terms.
    organism (str): Organism database to use (e.g., "org.Hs.eg.db").
    ont (str): Ontology to use (BP, MF, or CC).
    
    Returns:
    pd.DataFrame: Original DataFrame with an additional column for Wang similarity.
    """
    try:
        # Check if 'ID' column exists
        if 'ID' not in dataframe.columns:
            raise ValueError("The DataFrame must contain a column named 'ID' with GO terms.")

        # Extract GO terms
        go_terms = dataframe['ID'].dropna().unique()
        if len(go_terms) < 2:
            # If fewer than 2 terms, similarity is not meaningful
            dataframe['wang_similarity'] = 1.0
            return dataframe

        # Convert GO terms to R vector
        r_go_terms = robjects.StrVector(go_terms)

        # Load necessary R libraries
        robjects.r(f'''
            library(GOSemSim)
            go_db <- godata(annoDb = "{organism}", 
                            ont = "{ont}", 
                            computeIC = TRUE)
        ''')

        # Assign GO terms in R and calculate similarity matrix
        robjects.r.assign("go_terms", r_go_terms)
        robjects.r('''
            sim_matrix <- mgoSim(GO1 = go_terms, 
                                 GO2 = go_terms, 
                                 semData = go_db, 
                                 measure = "Wang", 
                                 combine = NULL)
            sim_matrix[is.na(sim_matrix)] <- 0  # Replace NA with 0
            mean_sims <- rowMeans(as.matrix(sim_matrix))  # Calculate mean similarity
            result_df <- data.frame(GO = go_terms, wang_similarity = mean_sims)
        ''')
        
        # Retrieve results as pandas DataFrame
        r_result_df = robjects.r('result_df')
        similarity_df = pandas2ri.rpy2py(r_result_df)

        # Merge similarities back to the original DataFrame
        dataframe = dataframe.merge(
            similarity_df,
            left_on='ID',
            right_on='GO',
            how='left'
        ).drop(columns=['GO'])

        return dataframe
    
    except Exception as e:
        print(f"Error calculating Wang similarity: {e}")
        dataframe['wang_similarity'] = np.nan
        return dataframe
