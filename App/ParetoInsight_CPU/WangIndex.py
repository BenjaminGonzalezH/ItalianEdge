######### Libraries #########
import numpy as np                                      # Efficient Math Operations.  
from gprofiler import GProfiler                         # Web-server for enrichment analisys.                                 
import pandas as pd                                     # Dataframe Managment.
from pygosemsim.similarity import wang                  # Wang index function.
from pygosemsim import graph                            # Create GoDag from source.
from pygosemsim import download                         # Obtain Go obo file.
from concurrent.futures import ProcessPoolExecutor      # Process managment.
from concurrent.futures import ThreadPoolExecutor       # Threads managment.
import networkx as nx                                   # Networks structures.
from pygosemsim import term_set

######### AUX elements. #########

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
    # Check input
    if not entrez_ids or not isinstance(entrez_ids, (list, np.ndarray)):
        raise ValueError("entrez_ids debe ser una lista no vacía de IDs.")
    
    valid_sources = {"GO:BP", "GO:CC", "GO:MF", "KEGG", "REAC", "CORUM", "HP"}
    if any(src not in valid_sources for src in Ontology):
        raise ValueError(f"Ontology contiene fuentes no soportadas: {Ontology}")
    
    # Activate GProfiler instance and return dataframes.
    gp = GProfiler(return_dataframe=True)

    try:
        resultado = gp.profile(
            organism=organism,                          # Species of study.
            query=entrez_ids,                           # EntrezID provided in input.
            no_evidences= False,                        # No use experimental evidence.
            user_threshold=1.0,                         # No stadistic filter.
            sources=Ontology                            # Source: GO:BP for example.
        )
    except Exception as e:
        raise RuntimeError(f"Error in Gprofiler query: {e}")

    if resultado.empty:
        return {}

    # Obtain terms for entrezIDs.
    exploded = resultado[['intersections', 'native']].explode('intersections')
    # Group all terms in a list per gene.
    Gene_to_go = exploded.groupby('intersections')['native'].agg(list).to_dict()

    return Gene_to_go

def calcular_batch(batch, preprocessed_terms, EntrezID, Gograph):
    results = []
    cache = {}
    for i, j in batch:
        terms_i = preprocessed_terms.get(EntrezID[i], [])
        terms_j = preprocessed_terms.get(EntrezID[j], [])
        if terms_i and terms_j:
            def sem_sim(t1, t2):
                key = (t1, t2) if t1 <= t2 else (t2, t1)
                if key not in cache:
                    cache[key] = wang(Gograph, t1, t2)
                return cache[key]
            score = term_set.sim_bma(terms_i, terms_j, sem_sim)
        else:
            score = 0.0
        results.append((i, j, score))
    return results

def WangIndexMatrix_1(EntrezID: list[str], 
                    organism: str = 'hsapiens',
                    Ontology: list[str] = ['GO:BP'], 
                    n_Process: int = 4,
                    download_f: bool = True) -> np.ndarray:
    """
    WangIndexMatrix(function): Computes Wang similarity matrix among genes using term_set.sim_bma.

    Parameters:
    - EntrezID: List of Entrez identifiers.
    - organism: Specie of study.
    - Ontology: Source of gprofiler query.
    - n_Process: Number of processes to use.

    Return:
    - WangSimilarity: Matrix with Wang index for every pair of genes.
    """
    # Obtener anotaciones GO para cada gen
    Dict_gene_Goterms = AnnotationFromEntrezIDs(EntrezID, Ontology, organism)
    
    # Descarga y gestión de archivos GO si es necesario
    if download_f:
        try:
            download.clear()
            download.obo("go-basic")
        except Exception as e:
            raise RuntimeError(f"Error in download GO OBO: {e}")

    # Cargar el grafo GO
    Gograph = graph.from_resource("go-basic")

    # Filtrar términos presentes en el grafo GO
    preprocessed_terms = {
        gene: [t for t in Dict_gene_Goterms.get(gene, []) if t in Gograph]
        for gene in EntrezID
    }

    # Crear matriz de salida
    n = len(EntrezID)
    WangSimilarity = np.zeros((n, n), dtype=np.float64)

    # Generar pares y dividir en batches
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    batch_size = max(1, len(pairs) // (n_Process * 2))
    batches = [pairs[k:k+batch_size] for k in range(0, len(pairs), batch_size)]

    # Multiprocesamiento
    with ProcessPoolExecutor(max_workers=n_Process) as executor:
        futures = [
            executor.submit(calcular_batch, batch, preprocessed_terms, EntrezID, Gograph)
            for batch in batches
        ]
        for future in futures:
            for i, j, score in future.result():
                WangSimilarity[i, j] = score
                WangSimilarity[j, i] = score  # Simetría

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