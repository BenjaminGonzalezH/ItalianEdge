######### Libraries #########
import numpy as np                                      # Efficient Math Operations.  
from gprofiler import GProfiler                         # Web-server for enrichment analisys.                                 
import pandas as pd                                     # Dataframe Managment.
from pygosemsim.similarity import wang                  # Wang index function.
from pygosemsim import graph                            # Create GoDag from source.
from functools import lru_cache                         # Use cached versions of functions.
from concurrent.futures import ProcessPoolExecutor      # Process managment.
from concurrent.futures import ThreadPoolExecutor       # Threads managment.
import networkx as nx                                   # Networks structures.

######### AUX elements. #########

@lru_cache(maxsize=None)
def cached_wang(godag: nx.DiGraph, 
                t1:str, 
                t2:str):
    """
    cached_wang(function): Cached version of wang calculus for terms. Avoid repetitive calculus.
    
    Parameters:
    - godag: Graph generated from pygosemsim functions (from_resourse).
    - t1 & t2: Go terms -> GO:<number> format.
    """
    return wang(godag, t1, t2)

@lru_cache(maxsize=None)
def cached_wang_similarity(terms_i: list[str], 
                           terms_j: list[str], 
                           Gograph: nx.DiGraph):
    """
    cached_wang_similarity(function): Cached version of wang calculus for terms collection. 
    Avoid repetitive calculus.
    
    Parameters:
    - terms_i & terms_j: Go terms -> GO:<number> format.
    - Gograph: Graph generated from pygosemsim functions (from_resourse).
    """
    return WangSimilarityPair(terms_i, terms_j, Gograph)

######### Functions #########

"""
This block contains all main functions.
"""

def AnnotationFromEntrezIDs(entrez_ids:list[np.str_],
                            Ontology=["GO:BP", "GO:CC", "GO:MF"], 
                            organism='hsapiens') -> dict[str , list[str]]:
    """
    AnnotationFromEntrezIDs(function): Use gprofiler to obtain all terms asociated with each entrezID.
    
    Parameters:
    - entrez_ids: List of Entrez identifiers.
    - Ontology: Source of gprofiler query.
    - organism: Specie of study.

    Returns:
    - gene_to_go: Dictionary that allocates entrez ID with their associated terms.
    """
    # Activate GProfiler instance and return dataframes.
    gp = GProfiler(return_dataframe=True)

    resultado = gp.profile(
        organism=organism,                          # Species of study.
        query=entrez_ids,                           # EntrezID provided in input.
        no_evidences= False,                        # No use experimental evidence.
        user_threshold=1.0,                         # No stadistic filter.
        sources=Ontology                            # Source: GO:BP for example.
    )

    if resultado.empty:
        return {}

    # Obtain terms for entrezIDs.
    exploded = resultado[['intersections', 'native']].explode('intersections')
    # Group all terms in a list per gene.
    Gene_to_go = exploded.groupby('intersections')['native'].agg(list).to_dict()

    return Gene_to_go

def WangSimilarityPair(
        terms1: list[str], 
        terms2: list[str], 
        godag: nx.DiGraph):
    """
    WangSimilarityPair(function): Calculates wang distanse betwenn two terms collections.
    
    Parameters:
    - terms1 & terms2: Go terms -> GO:<number> format.
    - Gograph: Graph generated from pygosemsim functions (from_resourse).
    """
    if not terms1 or not terms2:
        return 0.0
    
    # Calculate all pairwise similarities once (cached version of wang).
    similarities = {}
    for t1 in terms1:
        for t2 in terms2:
            similarities[(t1, t2)] = cached_wang(godag, t1, t2)
    
    # Calculate best matches (BMA criteria).
    scores1 = [max([similarities[(t1, t2)] for t2 in terms2], default=0) for t1 in terms1]
    scores2 = [max([similarities[(t1, t2)] for t1 in terms1], default=0) for t2 in terms2]
    return (sum(scores1) + sum(scores2)) / (len(scores1) + len(scores2))

def calcular_batch(batch: list[tuple[int, int]], 
                   preprocessed_terms: dict[str, list[str]], 
                   EntrezID: list[str], 
                   Gograph: nx.DiGraph):
    """
    calcular_batch(function): Compute Wang similarity for a batch of pairs.
    
    Parameters:
    - batch: Pair of index (genes) to process in the actual process.
    - preprocessed_terms: Terms filter by prescence in godag.
    - Gograph: Graph generated from pygosemsim functions (from_resourse).

    Return:
    - wang index of genes processed.
    """
    return [(i, j, cached_wang_similarity(tuple(preprocessed_terms[EntrezID[i]]),
                                          tuple(preprocessed_terms[EntrezID[j]]), Gograph)) for i, j in batch]

def WangIndexMatrix(EntrezID: list[str], 
                    organism: str = 'hsapiens',
                    Ontology: list[str] = ['GO:BP'], 
                    n_Process: int = 4) -> np.ndarray:
    """
    WangIndexMatrix(function): Computes Wang similarity matrix among genes.
    
    Parameters:
    - entrez_ids: List of Entrez identifiers.
    - organism: Specie of study.
    - Ontology: Source of gprofiler query.
    - n_Process: Amount of process to create.

    Return:
    - WangSimilarity: Matrix with wang index of every pair of genes.
    """
    # Obtain annotation from entrez IDs.
    Dict_gene_Goterms = AnnotationFromEntrezIDs(EntrezID, Ontology, organism)
    # Create Go DAG from go.obo downloaded previosly.
    Gograph = graph.from_resource("go")

    # Check if obtained terms are in the GO DAG.
    preprocessed_terms = {gene: [t for t in Dict_gene_Goterms.get(gene, []) if t in Gograph] 
                          for gene in EntrezID}

    # Create output matrix.
    n = len(EntrezID)
    WangSimilarity = np.zeros((n, n), dtype=np.float64)

    # Create pairs (upper triangular) and divide them into batch for every process.
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    batch_size = max(1, len(pairs) // (n_Process * 2))
    batches = [pairs[k:k+batch_size] for k in range(0, len(pairs), batch_size)]

    # Multiprocessing.
    with ProcessPoolExecutor(max_workers=n_Process) as executor:
        futures = [executor.submit(calcular_batch, batch, preprocessed_terms, EntrezID, Gograph) for batch in batches]

        # Merge results.
        for future in futures:
            for i, j, score in future.result():
                WangSimilarity[i, j] = score
                WangSimilarity[j, i] = score  # Ensure symmetry.

    return WangSimilarity

def Solution_Wang_index_similarity_Python(
        ids: list[str],
        similarity_matrix: np.ndarray,
        df: pd.DataFrame,
        groups_structure: list[set],
        num_threads: int = 4):
    """
    Builds a similarity matrix between solutions using Wang distance among genes.

    Parameters:
    - ids: Gene IDs used in similarity_matrix.
    - similarity_matrix: Precomputed Wang similarity matrix between genes.
    - df: DataFrame with ['Solution 1', 'Solution 2', 'Cluster 1', 'Cluster 2', 'Jaccard Similarity'].
    - groups_structure: Matrix from 'SolutionClusterMatrix'.
    - num_threads: Threads to use (default: 4).

    Returns:
    - final_matrix: Matrix with solution similarity scores.
    """
    # Define representative gene_id of solutions and output matrix.
    id_to_idx = {id_: idx for idx, id_ in enumerate(ids) if id_ != 'NA'}
    n = len(groups_structure)
    final_matrix = np.zeros((n, n), dtype=np.float32)

    # Structure for process managment.
    hashable_groups = [[frozenset(group) for group in cluster] for cluster in groups_structure]

    # Convert DataFrame values to expected types.
    df['Solution 1'] = df['Solution 1'].astype(int)
    df['Solution 2'] = df['Solution 2'].astype(int)
    df['Cluster 1'] = df['Cluster 1'].astype(int)
    df['Cluster 2'] = df['Cluster 2'].astype(int)
    df['Jaccard Similarity'] = df['Jaccard Similarity'].astype(float)

    #################################################################### Process row function for concurrency execution.
    def process_row(row):
        # Extract element from row and define output matrix.
        group_i, group_j, elem_i, elem_j, similarity = row
        local_matrix = np.zeros((n, n), dtype=np.float32)

        # Elements into range.
        if group_i >= n or group_j >= n or elem_i >= len(hashable_groups[group_i]) or elem_j >= len(hashable_groups[group_j]):
            return local_matrix
        
        # No comparison posible of groups.
        intersection = hashable_groups[group_i][elem_i] & hashable_groups[group_j][elem_j] - {'NA'}
        if len(intersection) < 2:
            return local_matrix

        # No enough genes.
        gene_indices = [id_to_idx[gene] for gene in intersection if gene in id_to_idx]
        if len(gene_indices) < 2:
            return local_matrix
        
        # Submatrix of wang index (upper triangle).
        sim_submatrix = similarity_matrix[np.ix_(gene_indices, gene_indices)]
        triu_values = sim_submatrix[np.triu_indices_from(sim_submatrix, k=1)]

        # If there is no index, just return a zero matrix.
        if triu_values.size == 0 or np.isnan(triu_values).all():
            return local_matrix

        # Acum index values.
        weighted_similarity = np.nanmean(triu_values) * similarity
        local_matrix[group_i, group_j] += weighted_similarity
        local_matrix[group_j, group_i] += weighted_similarity
        return local_matrix
    ################################################################################################################# Execute in concurrency.
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_row, df[['Solution 1', 'Solution 2', 'Cluster 1', 'Cluster 2', 'Jaccard Similarity']].itertuples(index=False, name=None)))

    # Merge solutions.
    for local_matrix in results:
        final_matrix += local_matrix

    # Normalization and fill diag with 1's.
    if np.max(final_matrix) > 0:
        final_matrix = final_matrix / np.max(final_matrix)
    np.fill_diagonal(final_matrix, 1)

    return final_matrix