######### Libraries #########
import go3                                              # Semantic similarities among genes or terms.
import requests                                         # Web request handler.
import gzip                                             # Compresed files managment.
import shutil                                           # Files managment.
import os                                               # OS callings.
from itertools import combinations                      # Possible pair combinations.
import pandas as pd                                     # Dataframe Managment.
import numpy as np                                      # Efficient Math Operations.
from concurrent.futures import ThreadPoolExecutor       # Threads managment.
from gprofiler import GProfiler                         # Web-server for enrichment analisys. 

######### AUX elements. #########

# URLs for species (more common in studies).
GAF_URL = {
    'goa_human':   "http://current.geneontology.org/annotations/goa_human.gaf.gz",
    'mgi':        "http://current.geneontology.org/annotations/mgi.gaf.gz",
    'fb':         "http://current.geneontology.org/annotations/fb.gaf.gz",
    'zfin':      "http://current.geneontology.org/annotations/zfin.gaf.gz",
    'sgd':        "http://current.geneontology.org/annotations/sgd.gaf.gz",
    'tair':     "http://current.geneontology.org/annotations/tair.gaf.gz",
    'pombase': "http://current.geneontology.org/annotations/pombase.gaf.gz",
    'bta':      "http://current.geneontology.org/annotations/goa_cow.gaf.gz",
    'goa_dog':  "http://current.geneontology.org/annotations/goa_dog.gaf.gz",
    'goa_pig':  "http://current.geneontology.org/annotations/goa_pig.gaf.gz",
    'goa_chicken': "http://current.geneontology.org/annotations/goa_chicken.gaf.gz",
    'rgd':      "http://current.geneontology.org/annotations/rgd.gaf.gz",
    'wb':       "http://current.geneontology.org/annotations/wb.gaf.gz"
}

######### Functions #########

"""
This block contains all main functions.
"""

def build_gene_mappings(gaf_file: str):
    """
    build_gene_mappings(function): Build dictionaries mapping IDs 
    to gene symbols.

    Parameters:
     - gaf_file: filename associated with the graph.

    Return:
        - Two dictionaries who associates genes id into database with their
        respective symbol.
    """
    id_to_symbol = {}
    symbol_to_id = {}

    with open(gaf_file+".gaf", "r") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) > 3:
                gene_id = parts[1]
                gene_symbol = parts[2]
                id_to_symbol[gene_id] = gene_symbol
                symbol_to_id[gene_symbol] = gene_id
    
    return id_to_symbol, symbol_to_id

def map_genes(genes, gaf_file: str, to: str = "symbol"):
    """
    map_genes(function): Transform a list of genes using the GAF file.
    
    Parameters:
        - genes: list of genes (can be IDs or symbols).
        - gaf_file: unzipped GAF file.
        - to: “symbol” → returns symbols, “id” → returns IDs.
    """
    id_to_symbol, symbol_to_id = build_gene_mappings(gaf_file)
    mapped = []

    for g in genes:
        if to == "symbol":
            mapped.append(id_to_symbol.get(g, g))
        elif to == "id":
            mapped.append(symbol_to_id.get(g, g))
        else:
            raise ValueError("El parámetro 'to' debe ser 'symbol' o 'id'")
    
    return mapped

def DownloadGAF(
        url: str, 
        output_filename: str
        ) -> None:
    """
    DownloadGAF(function): Download and descompress .gaf from a specie. Use the global variable
    GAF_URL to obtain correct url.

    Parameters:
        - url: Link to file.
        - output_filename: Name of the descompress file, it is not necessary to use ".gaf".
    """
    try:
        # Download file.
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save temporal compressed file.
        temp_gz = f"{output_filename}.gz"
        with open(temp_gz, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Descompres.
        print(f"Uncompressing: {temp_gz}")
        with gzip.open(temp_gz, 'rb') as f_in:
            with open(output_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove .gz file.
        os.remove(temp_gz)
        print(f"File ready: {output_filename + ".gaf"}")
        
        return True
        
    except Exception as e:
        print(f"Runtime Error {url}: {e}")
        return False

def pairs_to_matrix(gene_pairs, scores, genes):
    """
    pairs_to_matrix(function): Convert gene pairs + score list into a symmetric matrix.
    
    Parameters:
        - gene_pairs: list of tuples (gene1, gene2)
        - scores: list of floats (similarities in the same order as gene_pairs)
        - genes: list of genes
    
    Returns:
        - DataFrame with similarity matrix
    """
    df = pd.DataFrame(index=genes, columns=genes, dtype=float)

    for g in genes:
        df.loc[g, g] = 1.0

    for (g1, g2), score in zip(gene_pairs, scores):
        df.loc[g1, g2] = score
        df.loc[g2, g1] = score

    return df.to_numpy()

def SimilarityIndexMatrix(genes: str,
                    gaf_name: str,
                    ontology: str = "BP",
                    measure: str = "wang",
                    groupwise: str = "bma",
                    download_gaf: bool = True,
                    transform: bool = True,
                    load_go_terms: bool = True):
    """
    SimilarityIndexMatrix(function): Creates a similarity index matrix to compare genes
    according to biological information.

    Parameters:
        - genes: List of genes symbols.
        - gaf_name: Gaf file to use for comparison (has to be the correct specie).
        - ontology: Subontology from gene ontology.
        - similarity: Similarity measure (wang, lin, jc, simrel, iccoef, graphic, wang, topoicsim).
        - groupwise: Combination method to generate the similarities between genes (“bma” or “max”).
        - download_gaf: Flag to indicate to download gaf.
        - transform: Transform symbol according to graph.
        - load_go_terms: Download gene ontology terms.
    """
    try:
        # Input checking.
        if not genes or not isinstance(genes, (list, str)):
            raise TypeError("'genes' must be a list of strings")
        if not isinstance(gaf_name, str) or len(gaf_name.strip()) == 0:
            raise TypeError("'gaf_name' must be a valid string")
        if ontology not in ["BP", "MF", "CC"]:
            raise ValueError("'ontology' only supports 'BP', 'MF' o 'CC'.")
        if groupwise not in ["bma", "max"]:
            raise ValueError("'groupwise' only supports 'bma' o 'max'.")

        # Download gaf, transform genes symbols and load gene ontology terms
        # if it is neccesary.
        if download_gaf:
            if gaf_name not in GAF_URL:
                print(GAF_URL)
                raise ValueError("gaf_name is not in GAF_URL")
            DownloadGAF(GAF_URL[gaf_name], gaf_name)
        if transform:
            genes = map_genes(genes, gaf_name+".gaf", to="symbol")
        if load_go_terms:
            go3.load_go_terms()
        
        # Wang index calculus.
        annotations = go3.load_gaf(gaf_name)
        counter = go3.build_term_counter(annotations)
        gene_pairs = list(combinations(genes, 2))
        scores = go3.compare_gene_pairs_batch(gene_pairs, ontology, measure, groupwise, counter)
    
    # Exception handler.
    except Exception as e:
        raise RuntimeError(f"Error in find_equivalent_clusters: {e}")
    else:
        return pairs_to_matrix(gene_pairs, scores, genes)

def Solution_Wang_index_similarity_Python(
        ids: list[str],
        similarity_matrix: np.ndarray,
        df: pd.DataFrame,
        groups_structure: list[set],
        num_threads: int = 4) -> np.ndarray:
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