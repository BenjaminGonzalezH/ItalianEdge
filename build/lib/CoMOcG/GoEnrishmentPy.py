######### Libraries #########
from gprofiler import GProfiler
import mygene
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

######### AUX elements. #########

# URLs for species (more common).
GAF_URLS = {
    'human':     ('http://current.geneontology.org/annotations/goa_human.gaf.gz', 9606),
    'mouse':     ('http://current.geneontology.org/annotations/mgi.gaf.gz',       10090),
    'fly':       ('http://current.geneontology.org/annotations/fb.gaf.gz',        7227),
    'zebrafish': ('http://current.geneontology.org/annotations/zfin.gaf.gz',      7955),
    'yeast':     ('http://current.geneontology.org/annotations/sgd.gaf.gz',        559292),
    'arabidopsis': ('http://current.geneontology.org/annotations/tair.gaf.gz',    3702),
}

######### Functions #########

def chunks(lst, n):
    """Divide a list into n size blocks"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def query_mygene_block(
        block: list[str], 
        scopes: list[str], 
        taxID: int)->pd.DataFrame:
    """
    query_mygene_block (function): Do a request to MyGene servers of a block of 100 genes symbols
    using a taxonomy identifier and scopes (types) of symbols.

    Parameters:
    - block (list[str]): List of gene symbols.
    - scopes (list[str]): Symbol to process for mygene.
    - taxID (int): Taxonomy identifier for mygene.info.

    Returns:
    - Dataframe with entrez ID's.
    """
    mg = mygene.MyGeneInfo()
    try:
        results = mg.querymany(block, scopes=scopes, fields='entrezgene', species=taxID, as_dataframe=True)
        if results.empty:
            return pd.DataFrame()

        # 'notfound' Managment.
        if 'notfound' not in results.columns:
            results['notfound'] = False
        else:
            results['notfound'] = results['notfound'].fillna(False)

        # Take valids.
        if 'entrezgene' in results.columns:
            valid = results[(~results['notfound']) & (results['entrezgene'].notnull())]
        else:
            valid = pd.DataFrame()

        return valid

    except requests.exceptions.RequestException as e:
        print(f"Error en bloque MyGene.info: {e}")
        return pd.DataFrame()

def convert_symbols_to_entrez_Python(
        symbol_list: list[str], 
        organism_gp:str= 'hsapiens', 
        taxID: int= 9606, 
        scopes_mg: list[str]= ['symbol', 'alias', 'tair', 'accession'], 
        na_value: str= 'NA', 
        threads: int= 4)->list[str]:
    """
    convert_symbols_to_entrez_Python (function): Convert a list of various ID's into EntrezID's using
    GProfiler and MyGene.info.

    Parameters:
    - symbol_list (list[str]): List of gene symbols.
    - organims (str): Organims name for gene profiler.
    - taxID (int): Taxonomy identifier for mygene.info.
    - scopes_mg (list[str]): Symbol to process for mygene.
    - na_value (str): Value for NA or non-valid results.
    - threads (int): number of threads to use. 
    """
    if not isinstance(symbol_list, list) or len(symbol_list) == 0:
        raise ValueError("empty list.")

    conversion_dict = {}
    unmapped = symbol_list.copy()

    # --- gProfiler ---
    try:
        # Call GProfiler.
        gprof = GProfiler(return_dataframe=True)
        conversion = gprof.convert(organism=organism_gp, query=symbol_list, target_namespace='ENTREZGENE_ACC')

        # Take valid_conversions.
        valid_conversions = conversion[conversion['converted'].notnull()]
        valid_conversions = valid_conversions[valid_conversions['converted'].apply(lambda x: str(x).isnumeric())]
        valid_conversions['converted'] = valid_conversions['converted'].astype(int)

        # Configure to use just the minor EntrezID.
        grouped = valid_conversions.groupby('incoming').agg({'converted': 'min'}).reset_index()
        conversion_dict = dict(zip(grouped['incoming'], grouped['converted'].astype(str)))

        # Check non-maping genes.
        mapped_genes = set(conversion_dict.keys())
        unmapped = [gene for gene in symbol_list if gene not in mapped_genes]
        print(f"gProfiler → {len(mapped_genes)} transformed, {len(unmapped)} no match.")

    except requests.exceptions.RequestException as e:
        print(f"gProfiler connection error.: {e}")
        print("Continue with no results from gene profiler...")

    # --- MyGene.info threads ---
    if unmapped:
        print(f"Consultando MyGene.info para {len(unmapped)} genes...")
        mg_valid_frames = []
        blocks = list(chunks(unmapped, 1000))

        # calls my gene info using blocks of 1000 genes.
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(query_mygene_block, block, scopes_mg, taxID) for block in blocks]
            for future in as_completed(futures):
                result = future.result()
                if not result.empty:
                    mg_valid_frames.append(result)

        # Merge all valid results.
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

            # Print no obtained entrezID.
            still_unmapped = set(unmapped) - set(mg_mapping.keys())
            if still_unmapped:
                print(f"Warning: {len(still_unmapped)} entez no found by MyGene.info.")
        else:
            print("No valid results obtained from MyGene.info.")

    # --- Build list checking the original order ---
    entrez_ids = [conversion_dict.get(gene, na_value) for gene in symbol_list]

    return entrez_ids

def go_enrichment_entrez_Python(
        entrez_ids:list[str], 
        organism:str= 'hsapiens', 
        Ontology:list[str]= ['GO:BP'], 
        evidences:bool= False):
    """
    go_enrichment_entrez_Python (functio): Performs Go or other database enrichment using a list
    of EntrezID's.

    Parameters:
    - entrez_ids (list): List of genes (Entrez IDs).
    - organism (str): Organism (default: 'hsapiens').
    - Ontology (list[str]): Ontology and database to use for enrichment.
    - evidences (bool): Consider experimental evidence of terms.

    Retorna:
    - DataFrame con resultados de enriquecimiento GO ordenados por p-valor.
    """
    # Initialize g:Profiler
    gp = GProfiler(return_dataframe=True)

    # Execute terms analisys.
    results = gp.profile(
        organism=organism,
        query=entrez_ids,
        sources=Ontology,
        no_evidences=not evidences
    )

    if results.empty:
        print("No terms found.")
        return pd.DataFrame()

    # Order by p-valor
    results_sorted = results.sort_values('p_value').reset_index(drop=True)
    return results_sorted