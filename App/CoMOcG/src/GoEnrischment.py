# Importar librerías necesarias
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np                                                      # Efficient Math Operations.
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging

rpy2_logger.setLevel(logging.ERROR)  # Suprimir mensajes de R

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
    

def calculate_wang_distance_for_equivalent_clusters(equivalent_pairs_df: pd.DataFrame, wang_distance_df: pd.DataFrame, solutions: list[list[set]]) -> pd.DataFrame:
    """
    Calcula la distancia de Wang promedio entre los pares de genes en los clusters equivalentes.

    Parameters:
    - equivalent_pairs_df (pd.DataFrame): DataFrame con los pares de soluciones, los clusters equivalentes y las similitudes de Jaccard.
    - wang_distance_df (pd.DataFrame): DataFrame con las distancias de Wang entre genes (un DataFrame cuadrado indexado por los identificadores de genes).
    - solutions (list[list[set]]): Lista de soluciones, cada una representada por una lista de conjuntos de clústeres.

    Returns:
    - pd.DataFrame: Un DataFrame con los pares de clusters equivalentes y la distancia promedio de Wang entre los genes de esos pares.
    """
    all_wang_distances = []

    for _, row in equivalent_pairs_df.iterrows():
        # Obtener el par de soluciones y convertir los índices de solución correctamente
        solution_pair_str = row['Solution Pair']
        solution_pair = [int(s.split()[1]) for s in solution_pair_str.split(" vs ")]  # Extraer los índices (0, 1)

        equivalent_clusters = row['Equivalent Clusters']
        
        wang_distances = []
        
        # Para cada par de clusters equivalentes, obtener los genes de los clusters
        for cluster1, cluster2 in equivalent_clusters:
            # Asegurarse de que los índices de los genes sean enteros
            genes_cluster1 = list(solutions[solution_pair[0]][cluster1])  # Genes en el primer cluster de la solución 1
            genes_cluster2 = list(solutions[solution_pair[1]][cluster2])  # Genes en el segundo cluster de la solución 2

            # Calcular las distancias de Wang entre todos los pares de genes entre los dos clusters
            for gene1 in genes_cluster1:
                for gene2 in genes_cluster2:
                    print(type(gene1))
                    print(type(gene2))
                    wang_distances.append(wang_distance_df.loc[gene1, gene2])

        # Calcular el promedio de las distancias de Wang para estos clusters equivalentes
        if wang_distances:
            average_wang_distance = sum(wang_distances) / len(wang_distances)
        else:
            average_wang_distance = None

        all_wang_distances.append((solution_pair_str, average_wang_distance))

    wang_distance_df_result = pd.DataFrame(all_wang_distances, columns=["Solution Pair", "Average Wang Distance"])

    return wang_distance_df_result