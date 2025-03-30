# Importar librerías necesarias
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np                                                      # Efficient Math Operations.
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
from ast import literal_eval
import re
from concurrent.futures import ThreadPoolExecutor

rpy2_logger.setLevel(logging.ERROR)  # Suprimir mensajes de R.

pandas2ri.activate()

# Importar paquetes R de forma segura
def load_r_package(package_name):
    """Carga un paquete R si no está ya cargado."""
    if not robjects.r(f'"{package_name}" %in% loadedNamespaces()')[0]:
        robjects.r(f'library({package_name})')

# Cargar paquetes esenciales
load_r_package("AnnotationDbi")
load_r_package("biomaRt")
load_r_package("clusterProfiler")
load_r_package("GOSemSim")
load_r_package("AnnotationDbi")
load_r_package("BiocParallel")
load_r_package("parallel")

def convert_symbols_to_entrez(gene_symbols, organism="org.Hs.eg.db", keytype="SYMBOL"):
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

        # Cargar paquetes R necesarios
        load_r_package(organism)

        # Convertir a R vector
        r_genes = robjects.StrVector(gene_symbols)
        robjects.r.assign("gene_symbols", r_genes)

        # Código en R optimizado
        r_code = f"""
        entrez_ids <- mapIds(
            x = {organism}, 
            keys = gene_symbols, 
            column = "ENTREZID", 
            keytype = "{keytype}", 
            multiVals = "list"  # Obtener todos los posibles IDs
        )

        # Seleccionar el menor ID si hay múltiples
        entrez_ids <- lapply(entrez_ids, function(x) if (is.null(x)) NA else min(as.character(x)))
        entrez_ids <- unlist(entrez_ids)

        # Mantener estructura y evitar NA no manejables en Python
        entrez_ids[is.na(entrez_ids)] <- "NA"
        """
        robjects.r(r_code)

        # Obtener los IDs desde R
        entrez_ids = list(robjects.r('entrez_ids'))

        # Convertir todos los IDs a str en Python
        entrez_ids = [str(x) for x in entrez_ids]

        return entrez_ids

    except Exception as e:
        print(f"Error in convert_symbols_to_entrez: {e}")
        return []
    except Exception as e:
        print(f"Error in convert_symbols_to_entrez: {e}")
        return []

def perform_go_enrichment(gene_list, 
                          organism="org.Hs.eg.db", 
                          ont="BP", 
                          convert_ids=True,
                          keytype="SYMBOL"):
    """
    Perform GO enrichment analysis.
    """
    try:
        # Convertir IDs si es necesario
        if convert_ids:
            entrez_ids = convert_symbols_to_entrez(gene_list, organism, keytype)
            print(f"Converted {len(gene_list)} genes to {len(entrez_ids)} Entrez IDs.")
        else:
            entrez_ids = gene_list

        if not entrez_ids:
            raise ValueError("No valid Entrez IDs.")

        # Asignar genes en R
        r_genes = robjects.StrVector(entrez_ids)
        robjects.r.assign("gene_list", r_genes)

        # Ejecutar enriquecimiento GO
        r_code = f"""
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
        result_df = robjects.r(r_code)
        return pandas2ri.rpy2py(result_df)

    except Exception as e:
        print(f"Error in perform_go_enrichment: {e}")
        return pd.DataFrame()

def calculate_wang_distance_matrix(gene_list, organism="org.Hs.eg.db", ont="BP", convert_ids=True,
                                   keytype="SYMBOL"):
    """
    Calculate a matrix of Wang semantic distances for a list of genes.
    """
    try:
        # Validación de parámetros
        if gene_list is None or len(gene_list) < 2:
            raise ValueError("The gene list must contain at least two valid entries.")

        if organism is None:
            raise ValueError("Organism database cannot be None.")
        
        if ont is None:
            raise ValueError("Ontology type cannot be None. Use 'BP', 'MF', or 'CC'.")

        # Convertir IDs si es necesario
        if convert_ids:
            entrez_ids = convert_symbols_to_entrez(gene_list, organism, keytype)
            if not entrez_ids:
                raise ValueError("No valid Entrez IDs found after conversion.")
            print(f"Converted {len(gene_list)} genes to {len(entrez_ids)} Entrez IDs.")
        else:
            entrez_ids = gene_list

        # Convertir a R vector
        r_gene_list = robjects.StrVector(entrez_ids)
        robjects.r.assign("gene_list", r_gene_list)

        # Ejecutar código R con validación de parámetros
        r_code = f"""
        library(GOSemSim)
        go_db <- godata(annoDb = "{organism}", ont = "{ont}", computeIC = TRUE)
        sim_matrix <- mgeneSim(genes = gene_list, semData = go_db, measure = "Wang")
        sim_matrix[is.na(sim_matrix)] <- 0  # Replace NA with 0
        as.data.frame(sim_matrix)  # Convert to DataFrame
        """
        
        sim_matrix_df = robjects.r(r_code)

        # Convertir DataFrame de R a pandas
        sim_matrix_df = pandas2ri.rpy2py(sim_matrix_df)
        
        return sim_matrix_df

    except Exception as e:
        print(f"Error in calculate_wang_distance_matrix: {e}")
        return pd.DataFrame()
    
def calculate_wang_distance_matrix_1(gene_list, organism="org.Hs.eg.db", ont="BP", convert_ids=True,
                                 keytype="SYMBOL", num_cores=None):
    """
    Calculate a matrix of Wang semantic distances for a list of genes.
    Parameters:
        gene_list: List of genes to analyze
        organism: Organism database (default: "org.Hs.eg.db")
        ont: Ontology type ("BP", "MF", or "CC")
        convert_ids: Whether to convert gene symbols to Entrez IDs
        keytype: Type of gene ID if converting
        num_cores: Number of CPU cores to use (None for automatic detection)
    """
    try:
        # Validación de parámetros
        if not gene_list or len(gene_list) < 2:
            raise ValueError("The gene list must contain at least two valid entries.")

        # Pre-cargar paquetes R
        if not hasattr(robjects.r, 'GOSemSim'):
            robjects.r('suppressPackageStartupMessages(library(GOSemSim))')
        
        # Convertir IDs si es necesario
        if convert_ids:
            entrez_ids = convert_symbols_to_entrez(gene_list, organism, keytype)
            if not entrez_ids:
                raise ValueError("No valid Entrez IDs found after conversion.")
            print(f"Converted {len(gene_list)} genes to {len(entrez_ids)} Entrez IDs.")
        else:
            entrez_ids = gene_list

        with robjects.local_context() as rlc:
            # Configurar paralelización
            if num_cores is not None:
                robjects.r(f'''
                if (requireNamespace("parallel", quietly = TRUE)) {{
                    library(parallel)
                    options(mc.cores = {num_cores})
                    message("Using {num_cores} CPU cores")
                }} else {{
                    warning("parallel package not available, using single core")
                }}
                ''')
            else:
                # Detección automática (usa todos los cores menos 1)
                robjects.r('''
                if (requireNamespace("parallel", quietly = TRUE)) {
                    library(parallel)
                    options(mc.cores = max(1, detectCores() - 1))
                    message(paste("Using", getOption("mc.cores"), "CPU cores (auto-detected)"))
                }
                ''')

            # Preparar datos
            r_gene_list = robjects.StrVector(entrez_ids)
            robjects.r.assign("gene_list", r_gene_list)
            
            # Cachear godata si no existe
            if not hasattr(robjects.r, 'go_db'):
                robjects.r(f'go_db <- godata(annoDb = "{organism}", ont = "{ont}", computeIC = TRUE)')
            
            # Calcular matriz de similitud
            r_code = '''
            sim_matrix <- mgeneSim(genes = gene_list, 
                                  semData = go_db, 
                                  measure = "Wang")
            sim_matrix[is.na(sim_matrix)] <- 0
            as.data.frame(sim_matrix)
            '''
            
            sim_matrix_df = robjects.r(r_code)
            
            # Convertir a pandas eficientemente
            with (robjects.default_converter + pandas2ri.converter).context():
                sim_matrix_df = robjects.conversion.rpy2py(sim_matrix_df)
            
            return sim_matrix_df

    except Exception as e:
        print(f"Error in calculate_wang_distance_matrix: {e}")
        return pd.DataFrame()

def safe_literal_eval(s):
    """Evalúa strings que contienen estructuras de Python de forma segura, incluso con np.float64"""
    if isinstance(s, str):
        # Reemplazar np.float64 para que literal_eval pueda procesarlo
        s_clean = re.sub(r'np\.float64\(([^)]+)\)', r'\1', s)
        return literal_eval(s_clean)
    return s


def build_similarity_matrix(ids, similarity_matrix, df, groups_structure, num_threads=4):
    """
    Versión paralelizada de build_similarity_matrix usando múltiples hebras.
    
    Args:
        ids: Lista de IDs en el orden de similarity_matrix
        similarity_matrix: Matriz numpy de similitud
        df: DataFrame con ['Solution Pair', 'Equivalent Clusters']
        groups_structure: Lista de listas de conjuntos de IDs
        num_threads: Número de hebras a usar (por defecto 4)
    
    Returns:
        Matriz de similitud promediada (numpy array)
    """
    # Mapeo de ID a índice
    id_to_idx = {id_: idx for idx, id_ in enumerate(ids) if id_ != 'NA'}
    n = len(groups_structure)
    final_matrix = np.zeros((n, n))
    count_matrix = np.zeros((n, n))
    
    # Convertir a frozensets
    hashable_groups = [[frozenset(group) for group in cluster] for cluster in groups_structure]
    
    def parse_entry(entry):
        if isinstance(entry, str):
            return literal_eval(entry.replace('np.float64', ''))
        return entry
    
    # Parsear el DataFrame
    df = df.copy()
    df['Solution Pair'] = df['Solution Pair'].apply(parse_entry)
    df['Equivalent Clusters'] = df['Equivalent Clusters'].apply(parse_entry)
    
    # Función para procesar una fila
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
    
    # Procesamiento paralelo
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_row, [row for _, row in df.iterrows()]))
    
    # Combinar resultados
    for local_matrix, local_count in results:
        final_matrix += local_matrix
        count_matrix += local_count
    
    # Calcular promedios finales
    with np.errstate(divide='ignore', invalid='ignore'):
        final_matrix = np.divide(final_matrix, count_matrix)
        final_matrix[np.isnan(final_matrix)] = 0
    
    # Hacer simétrica y diagonal a 1
    final_matrix = (final_matrix + final_matrix.T) / 2
    np.fill_diagonal(final_matrix, 1)
    
    return final_matrix