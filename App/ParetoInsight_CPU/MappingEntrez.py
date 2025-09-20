######### Libraries #########
import numpy as np                                                # Efficient Math Operations.
from gprofiler import GProfiler                                   # Web-server for enrichment analisys.                                    
import mygene                                                     # Web-server for gene info.
import requests                                                   # Web request handler.
from concurrent.futures import ThreadPoolExecutor, as_completed   # Thread Administration.
import pandas as pd                                               # Dataframe Managment.

######### Functions #########

def chunks(lst, n):
    """
    chunks(function): Divide a gene list into n size blocks. Avoid
    mygene limits of genes to process.
    """
    # Generate a lazy chunk to wait until the others querys in
    # mygene are done.
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def query_mygene_chunk(
        chunk: list[np.str_], 
        scopes: list[np.str_], 
        taxID: int) -> pd.DataFrame:
    """
    query_mygene_chunk(function): Do a request to MyGene servers of a block of 100 genes symbols
    using a taxonomy identifier and scopes (types) of symbols.

    Parameters:
    - chunk: List of gene symbols.
    - scopes: Symbol to process for mygene.
    - taxID: Taxonomy identifier for mygene.info.

    Returns:
    - valid: Dataframe with entrez ID's.
    """
    # Activate mygene service instance.
    mg = mygene.MyGeneInfo()
    try:
        # Query: Process every symbol according of the specified scopes
        # into EntrezID and return a dataframe. Using only genes from
        # taxonomy ID specified.
        results = mg.querymany(chunk, 
                               scopes=scopes, 
                               fields='entrezgene', 
                               species=taxID, 
                               as_dataframe=True)
        
        # If there is no result from query returns dataframes.
        if results.empty:
            return pd.DataFrame()

        # 'notfound' Managment. The idea is standarize the output from
        # the query, creating 2 cases:
        #   'notfound' column no found -> Create the column with false values.
        #   'notfound' with "Na" values -> Replace them with false.
        if 'notfound' not in results.columns:
            results['notfound'] = False
        else:
            results['notfound'] = results['notfound'].fillna(False)

        # Take valids.
        # If there are results in dataframe -> Take only rows with
        # valid entrezID and succesfull query.
        if 'entrezgene' in results.columns:
            valid = results[(~results['notfound']) & (results['entrezgene'].notnull())]
        # Empty dataframe otherwise.
        else:
            valid = pd.DataFrame()

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error in chunk MyGene.info: {e}")
    else:
        return valid

def ConvertToEntrezID(
        symbol_list: list[np.str_], 
        organism_gp:str = 'hsapiens', 
        taxID: int= 9606, 
        scopes_mg: list[str] = ['symbol', 'alias', 'tair', 'accession', 'refseq'], 
        na_value: str = 'NA', 
        n_threads: int= 4) -> list[str]:
    """
    ConvertToEntrezID (function): Convert a list of various ID's into EntrezID's using
    GProfiler and MyGene.info.

    Parameters:
    - symbol_list: List of gene symbols.
    - organims: Organims name for gene profiler.
    - taxID: Taxonomy identifier for mygene.info.
    - scopes_mg: Symbol type to process for mygene.
    - na_value: Value for NA or non-valid results.
    - n_threads: number of threads to use. 
    """
    if not isinstance(symbol_list, list) or len(symbol_list) == 0:
        raise ValueError("empty list.")

    # Create a structure that is used to ensure similar order between input
    # list and convertion list.
    conversion_dict = {}
    unmapped = symbol_list.copy()

    # --------------------------------------------------------------------------- gProfiler.
    try:
        # Activate GProfiler instance and return dataframes.
        gprof = GProfiler(return_dataframe=True)
        # Query for convertion.
        conversion = gprof.convert(organism=organism_gp,                # Organis name.
                                   query=symbol_list,                   # Symbols to process.
                                   target_namespace='ENTREZGENE_ACC')   # EntrezID specification.

        # Take valid_conversions.
        #   Take not null convertions and check if they are numeric values and then transform it into
        #   integer values.
        valid_conversions = conversion[conversion['converted'].notnull()]
        valid_conversions = valid_conversions[valid_conversions['converted'].apply(lambda x: str(x).isnumeric())]

        # Configure to use just the minor EntrezID.
        grouped = valid_conversions.groupby('incoming').agg({'converted': 'min'}).reset_index()
        conversion_dict = dict(zip(grouped['incoming'], grouped['converted']))

        # Check non-maping genes.
        mapped_genes = set(conversion_dict.keys())
        unmapped = [gene for gene in symbol_list if gene not in mapped_genes]
        print(f"gProfiler → {len(mapped_genes)} transformed, {len(unmapped)} no match.")

    except requests.exceptions.RequestException as e:
        print(f"gProfiler connection error.: {e}")
        print("Continue with no results from gene profiler...")

    # ---------------------------------------------------------------------- MyGene.info threads
    if unmapped:
        print(f"MyGene.info query for {len(unmapped)} genes...")
        mg_valid_frames = []
        blocks = list(chunks(unmapped, 1000))

        # Calls my gene info using blocks of 1000 genes.
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(query_mygene_chunk, block, scopes_mg, taxID) for block in blocks]
            for future in as_completed(futures):
                result = future.result()
                if not result.empty:
                    mg_valid_frames.append(result)

        # Merge all valid results.
        if mg_valid_frames:
            mg_valid = pd.concat(mg_valid_frames)
            mg_mapping = {}

            for _, row in mg_valid.iterrows():
                entrez = row['entrezgene']
                if isinstance(entrez, list):
                    entrez = [int(e) for e in entrez if str(e).isnumeric()]
                    if entrez:
                        mg_mapping[row.name] = str(min(entrez))  # Minor.
                elif pd.notnull(entrez) and str(entrez).isnumeric():
                        mg_mapping[row.name] = str(entrez)

            print(f"MyGene.info → {len(mg_mapping)} genes converted.")
            conversion_dict.update(mg_mapping)

            # Print no obtained entrezID.
            still_unmapped = set(unmapped) - set(mg_mapping.keys())
            if still_unmapped:
                print(f"Warning: {len(still_unmapped)} entrez no found by MyGene.info.")
        else:
            print("No valid results obtained from MyGene.info.")

    # --- Build list checking the original order ---
    entrez_ids = [conversion_dict.get(gene, na_value) for gene in symbol_list]

    return entrez_ids
