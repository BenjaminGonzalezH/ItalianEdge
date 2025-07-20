######### Libraries #########
import numpy as np
from gprofiler import GProfiler
import pandas as pd
from pygosemsim.similarity import wang
from pygosemsim import graph
from pygosemsim import download
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
from multiprocessing import shared_memory
import math
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import Counter

######### AUX Functions #########

def _compute_pair(i, j, term_list, godag):
    sim = wang(godag, term_list[i], term_list[j])
    return i, j, sim

def precompute_term_similarity_parallel(terms: set[str], 
                                        godag: nx.DiGraph, 
                                        n_jobs: int = -1) -> tuple[dict, np.ndarray]:
    """
    Precomputa todas las similitudes Wang entre términos GO únicos, usando paralelización.
    """
    term_list = list(terms)
    n_terms = len(term_list)
    term_index = {term: i for i, term in enumerate(term_list)}
    sim_matrix = np.zeros((n_terms, n_terms), dtype=np.float32)

    # Todos los pares (i, j) con i ≤ j (para simetría)
    index_pairs = [(i, j) for i in range(n_terms) for j in range(i, n_terms)]

    # Paralelización
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_compute_pair)(i, j, term_list, godag)
        for i, j in tqdm(index_pairs, desc="Calculando similitudes Wang")
    )

    # Rellenar matriz
    for i, j, sim in results:
        sim_matrix[i, j] = sim_matrix[j, i] = sim

    return term_index, sim_matrix

def vectorized_bma(terms1: list[str], 
                  terms2: list[str], 
                  term_index: dict, 
                  sim_matrix: np.ndarray) -> float:
    """Cálculo vectorizado de Best Match Average."""
    if not terms1 or not terms2:
        return 0.0
    
    # Filtrado de términos válidos
    idx1 = [term_index[t] for t in terms1 if t in term_index]
    idx2 = [term_index[t] for t in terms2 if t in term_index]
    
    if not idx1 or not idx2:
        return 0.0
    
    # Cálculo matricial optimizado
    submatrix = sim_matrix[np.ix_(idx1, idx2)]
    max_sim1 = submatrix.max(axis=1).sum()
    max_sim2 = submatrix.max(axis=0).sum()
    
    return (max_sim1 + max_sim2) / (len(idx1) + len(idx2))

######### Core Functions #########

def AnnotationFromEntrezIDs(entrez_ids: list[str], 
                           Ontology: list[str] = ["GO:BP"], 
                           organism: str = 'hsapiens') -> dict[str, list[str]]:
    """Obtiene anotaciones GO con filtrado integrado."""
    gp = GProfiler(return_dataframe=True)
    
    try:
        result = gp.profile(
            organism=organism,
            query=entrez_ids,
            no_evidences=False,
            user_threshold=1.0,
            sources=Ontology
        )
    except Exception as e:
        raise RuntimeError(f"Error en GProfiler: {e}")

    if result.empty:
        return {}

    # Procesamiento optimizado con Pandas
    return (
        result.explode('intersections')
        .groupby('intersections')['native']
        .agg(list)
        .to_dict()
    )

def calcular_sim_score(i, j, preprocessed_terms, term_index, sim_matrix):
    terms_i = preprocessed_terms.get(str(i), [])
    terms_j = preprocessed_terms.get(str(j), [])
    score = vectorized_bma(terms_i, terms_j, term_index, sim_matrix)
    return i, j, score

def WangIndexMatrix(EntrezID: list[str], 
                           organism: str = 'hsapiens',
                           Ontology: list[str] = ['GO:BP'], 
                           n_jobs: int = -1,
                           download_f: bool = True) -> np.ndarray:
    if not EntrezID:
        raise ValueError("La lista de EntrezID no puede estar vacía")
    
    # Paso 1: Anotaciones
    gene_to_terms = AnnotationFromEntrezIDs(EntrezID, Ontology, organism)
    
    # Paso 2: Grafo GO
    if download_f:
        download.clear()
        download.obo("go-basic")
    godag = graph.from_resource("go-basic")
    
    # Paso 3: 
    all_terms = {t for terms in gene_to_terms.values() for t in terms if t in godag}
    term_index, sim_matrix = precompute_term_similarity_parallel(all_terms, godag, n_jobs=n_jobs)

    # Paso 4: Preprocesamiento
    preprocessed_terms = {gene: [t for t in terms if t in term_index] 
                          for gene, terms in gene_to_terms.items()}
    
    # Paso 5: Índices para comparar
    n = len(EntrezID)
    matrix = np.zeros((n, n), dtype=np.float32)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]

    # Paso 6: Paralelización con joblib
    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
        results = parallel(
            delayed(calcular_sim_score)(i, j, preprocessed_terms, term_index, sim_matrix)
            for i, j in pairs
        )

    # Paso 7: Construcción de matriz
    for i, j, score in results:
        matrix[i, j] = score
        if i != j:
            matrix[j, i] = score
    
    return matrix