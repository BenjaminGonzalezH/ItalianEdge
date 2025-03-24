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

# Testing libraries.
from gprofiler import GProfiler
import mygene
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pygosemsim import graph, SemanticSim, download_ontology
from bioservices import UniProt
from itertools import combinations

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


def chunks(lst, n):
    """Divide lista en bloques de tamaño n"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def query_mygene_block(block, scopes, taxID):
    mg = mygene.MyGeneInfo()
    try:
        results = mg.querymany(block, scopes=scopes, fields='entrezgene', species=taxID, as_dataframe=True)
        if results.empty:
            return pd.DataFrame()

        # Manejo 'notfound'
        if 'notfound' not in results.columns:
            results['notfound'] = False
        else:
            results['notfound'] = results['notfound'].fillna(False)

        # Filtrar válidos
        if 'entrezgene' in results.columns:
            valid = results[(~results['notfound']) & (results['entrezgene'].notnull())]
        else:
            valid = pd.DataFrame()

        return valid

    except requests.exceptions.RequestException as e:
        print(f"Error en bloque MyGene.info: {e}")
        return pd.DataFrame()

def convert_symbols_to_entrez_1(symbol_list, organism_gp='hsapiens', taxID=9606, 
                                scopes_mg=['symbol', 'alias', 'tair', 'accession'], na_value='NA', threads=4):
    """
    Convierte SYMBOLs a Entrez IDs usando gProfiler y MyGene.info, optimizada con hebras para bloques.
    """
    if not isinstance(symbol_list, list) or len(symbol_list) == 0:
        raise ValueError("Lista vacía.")

    conversion_dict = {}
    unmapped = symbol_list.copy()

    # --- Primera capa: gProfiler ---
    try:
        gprof = GProfiler(return_dataframe=True)
        conversion = gprof.convert(organism=organism_gp, query=symbol_list, target_namespace='ENTREZGENE_ACC')

        valid_conversions = conversion[conversion['converted'].notnull()]
        valid_conversions = valid_conversions[valid_conversions['converted'].apply(lambda x: str(x).isnumeric())]
        valid_conversions['converted'] = valid_conversions['converted'].astype(int)

        # Agrupar y tomar el menor EntrezID
        grouped = valid_conversions.groupby('incoming').agg({'converted': 'min'}).reset_index()
        conversion_dict = dict(zip(grouped['incoming'], grouped['converted'].astype(str)))

        # Identificar genes no mapeados
        mapped_genes = set(conversion_dict.keys())
        unmapped = [gene for gene in symbol_list if gene not in mapped_genes]
        print(f"gProfiler → {len(mapped_genes)} mapeados, {len(unmapped)} no mapeados.")

    except requests.exceptions.RequestException as e:
        print(f"Error de conexión con gProfiler: {e}")
        print("Continuando sin resultados de gProfiler...")

    # --- Segunda capa: MyGene.info paralelizado ---
    if unmapped:
        print(f"Consultando MyGene.info para {len(unmapped)} genes...")
        mg_valid_frames = []
        blocks = list(chunks(unmapped, 1000))

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(query_mygene_block, block, scopes_mg, taxID) for block in blocks]
            for future in as_completed(futures):
                result = future.result()
                if not result.empty:
                    mg_valid_frames.append(result)

        # Concatenar todos los resultados válidos
        if mg_valid_frames:
            mg_valid = pd.concat(mg_valid_frames)
            mg_mapping = {}

            for index, row in mg_valid.iterrows():
                entrez = row['entrezgene']
                if isinstance(entrez, list):
                    entrez = [int(e) for e in entrez if isinstance(e, (int, str)) and str(e).isnumeric()]
                    if entrez:
                        mg_mapping[row.name] = str(min(entrez))  # Menor
                elif pd.notnull(entrez):
                    if isinstance(entrez, (int, str)) and str(entrez).isnumeric():
                        mg_mapping[row.name] = str(entrez)

            print(f"MyGene.info → {len(mg_mapping)} genes mapeados.")
            conversion_dict.update(mg_mapping)

            # Avisar no encontrados
            still_unmapped = set(unmapped) - set(mg_mapping.keys())
            if still_unmapped:
                print(f"Advertencia: {len(still_unmapped)} genes no encontrados en MyGene.info.")
        else:
            print("No se obtuvieron resultados válidos desde MyGene.info.")

    # --- Reconstruir lista manteniendo orden ---
    entrez_ids = [conversion_dict.get(gene, na_value) for gene in symbol_list]

    return entrez_ids

def go_enrichment_entrez(entrez_ids, organism='hsapiens', Ontology=['GO:BP'], evidences=False):
    """
    Realiza análisis de enriquecimiento GO para una lista de Entrez IDs.

    Parámetros:
    - entrez_ids (list): Lista de genes (Enterez IDs).
    - organism (str): Organismo (default: 'hsapiens').

    Retorna:
    - DataFrame con resultados de enriquecimiento GO ordenados por p-valor.
    """
    # Inicializar g:Profiler
    gp = GProfiler(return_dataframe=True)

    # Ejecutar enriquecimiento solo para términos GO
    results = gp.profile(
        organism=organism,
        query=entrez_ids,
        sources=Ontology,
        no_evidences=not evidences
    )

    if results.empty:
        print("No se encontraron términos enriquecidos.")
        return pd.DataFrame()

    # Ordenar por p-valor
    results_sorted = results.sort_values('p_value').reset_index(drop=True)
    return results_sorted


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
