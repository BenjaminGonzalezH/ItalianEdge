# Testing libraries.
from gprofiler import GProfiler
import mygene
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pygosemsim import download
from bioservices import UniProt
from itertools import combinations
import os
import gzip
import requests
from itertools import product
from pygosemsim import graph, similarity, annotation
from goatools import obo_parser
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.semantic import TermCounts, semantic_similarity
import gzip
from functools import lru_cache
import warnings
import pandas as pd

# URLs por especie (puedes agregar más)
GAF_URLS = {
    'human':     ('http://current.geneontology.org/annotations/goa_human.gaf.gz', 9606),
    'mouse':     ('http://current.geneontology.org/annotations/mgi.gaf.gz',       10090),
    'fly':       ('http://current.geneontology.org/annotations/fb.gaf.gz',        7227),
    'zebrafish': ('http://current.geneontology.org/annotations/zfin.gaf.gz',      7955),
    'yeast':     ('http://current.geneontology.org/annotations/sgd.gaf.gz',        559292),
    'arabidopsis': ('http://current.geneontology.org/annotations/tair.gaf.gz',    3702),
}

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