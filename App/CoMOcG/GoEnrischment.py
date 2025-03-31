######### Libraries #########
import rpy2.robjects as robjects                                        # Mapping structures and code to R enviroment.
from rpy2.robjects import pandas2ri                                     # Dataframe transfer between Python & R.
import pandas as pd                                                     # Dataframe managment.
import numpy as np                                                      # Numerical math ADT.
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger         # Set message from R enviroment.
import logging
from ast import literal_eval                                            # Evaluate stings.
import re                                                               # Regular expressions.
from concurrent.futures import ThreadPoolExecutor                       # Threads Managment Interface.

# Configurations: 
rpy2_logger.setLevel(logging.ERROR)  # Allow only error messages.
pandas2ri.activate()                 # Activate interface for Dataframe transfer.

######### Functions #########

"""
This block contains all main functions.
"""

def load_r_package(
        package_name:str
        )->None:
    """
    load_r_package (function): load an pre-installed R Package from your version
    of R.

    Parameters:
    - package_name: Name of the package.
    """
    if not robjects.r(f'"{package_name}" %in% loadedNamespaces()')[0]:
        robjects.r(f'library({package_name})')

# Loading of essencial packages.
load_r_package("AnnotationDbi")
load_r_package("biomaRt")
load_r_package("clusterProfiler")
load_r_package("GOSemSim")

def convert_symbols_to_entrez(
        gene_symbols: list[str], 
        organism: str ="org.Hs.eg.db", 
        keytype: str ="SYMBOL"
        ) -> list[str]:
    """
    Convert gene symbols to Entrez IDs efficiently, returning a list of strings.

    Parameters:
    - gene_symbols (list): List of gene symbols.
    - organism (str): Organism database (default: "org.Hs.eg.db" for humans).

    Returns:
    - list: List of corresponding Entrez IDs as strings, preserving order.
    """
    try:
        if not gene_symbols or len(gene_symbols) == 0:
            raise ValueError("Gene symbol list is empty.")

        # Load organism database.
        load_r_package(organism)

        # Convert list of strings into a vector for R enviroment.
        r_genes = robjects.StrVector(gene_symbols)
        robjects.r.assign("gene_symbols", r_genes)

        r_code = f"""
        entrez_ids <- mapIds(
            x = {organism}, 
            keys = gene_symbols, 
            column = "ENTREZID", 
            keytype = "{keytype}", 
            multiVals = "list"  # Obtener todos los posibles IDs
        )

        # Select minus ID if there are many.
        entrez_ids <- lapply(entrez_ids, function(x) if (is.null(x)) NA else min(as.character(x)))
        entrez_ids <- unlist(entrez_ids)

        # NA if function did not found an Entrez ID.
        entrez_ids[is.na(entrez_ids)] <- "NA"
        """
        robjects.r(r_code)

        # Obtain list from R with EntrezIDs.
        entrez_ids = list(robjects.r('entrez_ids'))

        # Transform ID into strings.
        entrez_ids = [str(x) for x in entrez_ids]

        return entrez_ids

    except Exception as e:
        print(f"Error in convert_symbols_to_entrez: {e}")
        return []

def perform_go_enrichment(
        gene_list: list[str], 
        organism: str ="org.Hs.eg.db", 
        ont: str="BP", 
        convert_ids: bool =True,
        keytype: str="SYMBOL"
        )->pd.DataFrame:
    """
    perform_go_enrichment (function): Perform GO enrichment analysis.

    Parameters:
    - gene_list (list): List of Entrez ID's.
    - organism (str): Organism database (default: "org.Hs.eg.db" for humans).
    - ont (str): Ontology for study.
    - convert_ids (bool): Activate convertions.
    - keytype (str): Type of identifier that have gene list (default: "SYMBOL").

    Returns:
    - Enrichment_dataframe: Go terms with data.
    """
    try:
        # Convert ID's if it is necessary.
        if convert_ids:
            entrez_ids = convert_symbols_to_entrez(gene_list, organism, keytype)
            print(f"Converted {len(gene_list)} genes to {len(entrez_ids)} Entrez IDs.")
        else:
            entrez_ids = gene_list

        if not entrez_ids:
            raise ValueError("No valid Entrez IDs.")

        # Convert list of strings into a vector for R enviroment. 
        r_genes = robjects.StrVector(entrez_ids)
        robjects.r.assign("gene_list", r_genes)

        r_code = f"""
        # Load organism dataframe.
        library({organism})

        go_results <- enrichGO(
            gene = gene_list,
            OrgDb = {organism},
            ont = "{ont}",
            keyType = "ENTREZID",
            readable = TRUE
        )
        as.data.frame(go_results)
        """
        # Result into pandas dataframe.
        result_df = robjects.r(r_code)
        return pandas2ri.rpy2py(result_df)

    except Exception as e:
        print(f"Error in perform_go_enrichment: {e}")
        return pd.DataFrame()

def calculate_wang_distance_matrix(
        gene_list: list[str], 
        organism: str ="org.Hs.eg.db", 
        ont: str ="BP", 
        convert_ids: bool =True,
        keytype: str ="SYMBOL"
        ) -> pd.DataFrame:
    """
    calculate_wang_distance_matrix (function): Calculate a matrix of Wang semantic distances 
    for a list of genes.

    Parameters:
    - gene_list (list): List of Entrez ID's.
    - organism (str): Organism database (default: "org.Hs.eg.db" for humans).
    - ont (str): Ontology for study.
    - convert_ids (bool): Activate convertions.
    - keytype (str): Type of identifier that have gene list (default: "SYMBOL").

    Retun:
    - sim_matrix_df: Dataframe with EntrezID and matrix with distances.
    """
    try:
        # Check inputs.
        if gene_list is None or len(gene_list) < 2:
            raise ValueError("The gene list must contain at least two valid entries.")
        if organism is None:
            raise ValueError("Organism database cannot be None.")
        if ont is None:
            raise ValueError("Ontology type cannot be None. Use 'BP', 'MF', or 'CC'.")

        # Convert ID's if it is necessary.
        if convert_ids:
            entrez_ids = convert_symbols_to_entrez(gene_list, organism, keytype)
            if not entrez_ids:
                raise ValueError("No valid Entrez IDs found after conversion.")
            print(f"Converted {len(gene_list)} genes to {len(entrez_ids)} Entrez IDs.")
        else:
            entrez_ids = gene_list

        # Convert list of strings into a vector for R enviroment. 
        r_gene_list = robjects.StrVector(entrez_ids)
        robjects.r.assign("gene_list", r_gene_list)

        r_code = f"""
        # load Semantic distance dataframe.
        library(GOSemSim)

        go_db <- godata(annoDb = "{organism}", ont = "{ont}", computeIC = TRUE)
        sim_matrix <- mgeneSim(genes = gene_list, semData = go_db, measure = "Wang")
        sim_matrix[is.na(sim_matrix)] <- 0  # Replace NA with 0
        as.data.frame(sim_matrix)  # Convert to DataFrame
        """
        
        # Obtain dataframe with Wang distance.
        sim_matrix_df = robjects.r(r_code)
        sim_matrix_df = pandas2ri.rpy2py(sim_matrix_df)
        return sim_matrix_df

    except Exception as e:
        print(f"Error in calculate_wang_distance_matrix: {e}")
        return pd.DataFrame()

def safe_literal_eval(
        s:str
        ) -> str:
    """
    safe_literal_eval (function): Evaluate strings that contain Python structure (np.float64)
    in secure way.
    
    Parameters:
    - s: String that is going to be evaluate.
    """
    if isinstance(s, str):
        s_clean = re.sub(r'np\.float64\(([^)]+)\)', r'\1', s)
        return literal_eval(s_clean)
    return s

def build_similarity_matrix(
        ids: list[str], 
        similarity_matrix: np.ndarray, 
        df: pd.DataFrame, 
        groups_structure: list[set], 
        num_threads: int= 4):
    """
    build_similarity_matrix (function): Create a similary matrix between solutions using the wang
    distance among every gene.
    
    Parameters:
    - ids: IDs of genes that allocates sim_matrix_df.
    - similarity_matrix: Matrix with wang distance index from sim_matrix_df.
    - df: DataFrame con ['Solution Pair', 'Equivalent Clusters'] from function 'find_equivalent_clusters'
    - groups_structure: Matrix from 'SolutionClusterMatrix'.
    - num_threads: Threads to use (default: 4)
    
    Returns:
    - final_matrix: Matrix that allocate the distance matrix among solutions.
    """
    # Index Mapping for ID's.
    id_to_idx = {id_: idx for idx, id_ in enumerate(ids) if id_ != 'NA'}
    n = len(groups_structure)
    final_matrix = np.zeros((n, n))
    count_matrix = np.zeros((n, n))
    
    # Use frozensets (avoid modifications).
    hashable_groups = [[frozenset(group) for group in cluster] for cluster in groups_structure]
    
    def parse_entry(entry):
        if isinstance(entry, str):
            return literal_eval(entry.replace('np.float64', ''))
        return entry
    
    # Change of dataframe format.
    df = df.copy()
    df['Solution Pair'] = df['Solution Pair'].apply(parse_entry)
    df['Equivalent Clusters'] = df['Equivalent Clusters'].apply(parse_entry)
    
    # Process every row.
    def process_row(row):
        group_i, group_j = row['Solution Pair']
        equivalent_pairs = row['Equivalent Clusters']
        local_matrix = np.zeros((n, n))
        local_count = np.zeros((n, n))
        
        if group_i >= n or group_j >= n:
            return local_matrix, local_count
            
        sets_i = hashable_groups[group_i]
        sets_j = hashable_groups[group_j]
        
        for elem_i, elem_j in equivalent_pairs:
            if elem_i >= len(sets_i) or elem_j >= len(sets_j):
                continue
                
            set_i = sets_i[elem_i]
            set_j = sets_j[elem_j]
            intersection = set_i & set_j - {'NA'}
            
            if not intersection:
                continue
                
            sum_sim = 0
            count = 0
            intersection_list = list(intersection)
            
            for i in range(len(intersection_list)):
                for j in range(i, len(intersection_list)):
                    idx_a = id_to_idx.get(intersection_list[i], None)
                    idx_b = id_to_idx.get(intersection_list[j], None)
                    if idx_a is not None and idx_b is not None:
                        sum_sim += similarity_matrix[idx_a][idx_b]
                        count += 1
            
            if count > 0:
                avg_sim = sum_sim / count
                local_matrix[group_i][group_j] += avg_sim
                local_count[group_i][group_j] += 1
                local_matrix[group_j][group_i] += avg_sim
                local_count[group_j][group_i] += 1
                
        return local_matrix, local_count
    
    # Parallel processing.
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_row, [row for _, row in df.iterrows()]))
    
    # Combain results.
    for local_matrix, local_count in results:
        final_matrix += local_matrix
        count_matrix += local_count
    
    # final process.
    with np.errstate(divide='ignore', invalid='ignore'):
        final_matrix = np.divide(final_matrix, count_matrix)
        final_matrix[np.isnan(final_matrix)] = 0
    
    # Make it simmetric and diagonal.
    final_matrix = (final_matrix + final_matrix.T) / 2
    np.fill_diagonal(final_matrix, 1)
    
    return final_matrix