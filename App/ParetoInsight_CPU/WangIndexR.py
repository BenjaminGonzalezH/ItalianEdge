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
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from App.ParetoInsight_CPU.MappingEntrez import ConvertToEntrezID

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

def calculate_wang_distance_matrix(
        gene_list: list[str],
        organism: str = "org.Hs.eg.db", 
        organism_gp: str ="hsapiens",
        TaxID: int = 9606, 
        ont: str ="BP", 
        convert_ids: bool =True
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
            entrez_ids = ConvertToEntrezID(gene_list, organism_gp=organism_gp, taxID=TaxID)
            if not entrez_ids:
                raise ValueError("No valid Entrez IDs found after conversion.")
            print(f"Converted {len(gene_list)} genes to {len([id for id in entrez_ids if id != 'NA'])} Entrez IDs.")
        else:
            entrez_ids = gene_list

        # Convert list of strings into a vector for R enviroment. 
        r_gene_list = robjects.StrVector(entrez_ids)
        robjects.r.assign("gene_list", r_gene_list)

        r_code = f"""
        # load Semantic distance dataframe.
        library(GOSemSim)

        go_db <- godata(annoDb = "{organism}", ont = "{ont}")
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

def safe_literal_eval(s):
    """
    safe_literal_eval (function): Evaluate strings that contain Python structure (np.float64)
    in secure way.
    
    Parameters:
    - s: String that is going to be evaluated.
    
    Returns:
    - Evaluated Python object
    """
    if isinstance(s, str):
        s_clean = re.sub(r'np\.float64\(([^)]+)\)', r'\1', s)
        return literal_eval(s_clean)
    return s

def process_row_process(row_dict, hashable_groups, n, organism, ont, convert_ids, keytype):
    solution_pair = row_dict['Solution Pair']
    equivalent_pairs = row_dict['Equivalent Clusters']
    
    if not isinstance(solution_pair, tuple) or len(solution_pair) != 2:
        print(f"Warning: Solution Pair has unexpected format: {solution_pair}")
        return np.zeros((n, n), dtype=np.float32)
        
    group_i, group_j = solution_pair

    local_matrix = np.zeros((n, n), dtype=np.float32)
    if group_i >= n or group_j >= n:
        return local_matrix

    sets_i = hashable_groups[group_i]
    sets_j = hashable_groups[group_j]
    
    for elem_i, elem_j in equivalent_pairs:
        if elem_i >= len(sets_i) or elem_j >= len(sets_j):
            continue
        
        intersection = sets_i[elem_i] & sets_j[elem_j] - {'NA'}
        if len(intersection) < 2:
            continue
        
        DF_W = calculate_wang_distance_matrix(intersection, organism=organism, ont=ont, convert_ids=convert_ids, keytype=keytype)
        arr = DF_W.to_numpy()
        if arr.size == 0 or np.isnan(arr).all():
            local_matrix[group_i, group_j] += 0
            local_matrix[group_j, group_i] += 0
        else:
            similarity = np.nanmean(arr)
            local_matrix[group_i, group_j] += similarity
            local_matrix[group_j, group_i] += similarity

    return local_matrix

def process_row_process_unpack(args):
    return process_row_process(*args)

def Solution_Wang_index_similarity_rpy2(
        df: pd.DataFrame,
        groups_structure: list[set],
        num_threads: int = 4,
        organism: str ="org.Hs.eg.db", 
        ont: str ="BP", 
        convert_ids: bool =True,
        keytype: str ="SYMBOL"):
    
    n = len(groups_structure)
    final_matrix = np.zeros((n, n), dtype=np.float32)

    hashable_groups = [[frozenset(group) for group in cluster] for cluster in groups_structure]

    df = df.copy()
    df['Solution Pair'] = df['Solution Pair'].apply(safe_literal_eval)
    df['Equivalent Clusters'] = df['Equivalent Clusters'].apply(safe_literal_eval)
    rows = df[['Solution Pair', 'Equivalent Clusters']].to_dict('records')

    # Armar argumentos para pasar a cada proceso
    args = [(row, hashable_groups, n, organism, ont, convert_ids, keytype) for row in rows]

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(process_row_process_unpack, args)

    for local_matrix in results:
        final_matrix += local_matrix

    final_matrix = final_matrix / np.max(final_matrix)
    np.fill_diagonal(final_matrix, 1)
    
    return final_matrix

def Solution_Wang_index_similarity_Python(
        ids: list[str],
        similarity_matrix: np.ndarray,
        df: pd.DataFrame,
        groups_structure: list[set],
        num_threads: int = 4):
    """
    build_similarity_matrix (function): Create a similarity matrix between solutions using the wang
    distance among every gene.

    Parameters:
    - ids: IDs of genes that allocate similarity_matrix.
    - similarity_matrix: Precomputed Wang similarity matrix between genes.
    - df: DataFrame with ['Solution Pair', 'Equivalent Clusters'] from function 'find_equivalent_clusters'
    - groups_structure: Matrix from 'SolutionClusterMatrix'.
    - num_threads: Threads to use (default: 4)

    Returns:
    - final_matrix: Matrix that allocates the distance matrix among solutions.
    """
    id_to_idx = {id_: idx for idx, id_ in enumerate(ids) if id_ != 'NA'}
    n = len(groups_structure)
    final_matrix = np.zeros((n, n), dtype=np.float32)

    hashable_groups = [[frozenset(group) for group in cluster] for cluster in groups_structure]

    df = df.copy()
    df['Solution Pair'] = df['Solution Pair'].apply(safe_literal_eval)
    df['Equivalent Clusters'] = df['Equivalent Clusters'].apply(safe_literal_eval)

    rows = df[['Solution Pair', 'Equivalent Clusters']].to_dict('records')

    def process_row(row_dict):
        solution_pair = row_dict['Solution Pair']
        equivalent_pairs = row_dict['Equivalent Clusters']

        if not isinstance(solution_pair, tuple) or len(solution_pair) != 2:
            print(f"Warning: Solution Pair has unexpected format: {solution_pair}")
            return np.zeros((n, n), dtype=np.float32)

        group_i, group_j = solution_pair
        local_matrix = np.zeros((n, n), dtype=np.float32)

        if group_i >= n or group_j >= n:
            return local_matrix

        sets_i = hashable_groups[group_i]
        sets_j = hashable_groups[group_j]

        for elem_i, elem_j in equivalent_pairs:
            if elem_i >= len(sets_i) or elem_j >= len(sets_j):
                continue

            intersection = sets_i[elem_i] & sets_j[elem_j] - {'NA'}
            if len(intersection) < 2:
                continue

            gene_indices = [id_to_idx[gene] for gene in intersection if gene in id_to_idx]
            if len(gene_indices) < 2:
                continue

            # Extract submatrix and calculate the upper triangle mean (excluding diagonal)
            sim_submatrix = similarity_matrix[np.ix_(gene_indices, gene_indices)]
            triu_indices = np.triu_indices_from(sim_submatrix, k=1)
            triu_values = sim_submatrix[triu_indices]

            if triu_values.size == 0 or np.isnan(triu_values).all():
                continue

            similarity = np.nanmean(triu_values)
            local_matrix[group_i, group_j] += similarity
            local_matrix[group_j, group_i] += similarity

        return local_matrix

    # Ejecutar en paralelo
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_row, rows))

    for local_matrix in results:
        final_matrix += local_matrix

    if np.max(final_matrix) > 0:
        final_matrix = final_matrix / np.max(final_matrix)
    np.fill_diagonal(final_matrix, 1)

    return final_matrix