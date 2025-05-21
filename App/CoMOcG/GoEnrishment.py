######### Libraries #########
import numpy as np                   # Efficient Math Operations.
from gprofiler import GProfiler      # Web-server for enrichment analisys.                                    
import pandas as pd                  # Dataframe Managment.

######### AUX elements. #########

# URLs for species (more common in studies).
# My intention is provide a structure useful structure to check links for
# species gaf annotations. Also, taxonomy identifier for mygene petitions.
GAF_URLS = {
    'human':     ('goa_human', 9606),
    'mouse':     ('mgi',       10090),
    'fly':       ('fb',        7227),
    'zebrafish': ('zfin',      7955),
    'yeast':     ('sgd',        559292),
    'athaliana': ('tair',    3702),
    'schizosaccharomyces_pombe': ('pombase', 4896),
    'cow':       ('bta',     9913),
    'dog':       ('upa_dog', 9615),
    'pig':       ('upa_pig', 9823),
    'chicken':   ('upa_chicken', 9031),
    'rat':       ('rgd',     10116),
    'c_elegans': ('wb',      6239),
}

######### Functions #########

"""
This block contains all main functions.
"""

def GoEnrichment(
        entrez_ids:list[str], 
        organism:str = 'hsapiens', 
        Ontology:list[str] = ['GO:BP'], 
        evidences:bool = False):
    """
    GoEnrichment (functioN): Perform GO enrichment analysis using entrezID.

    Parameters:
    - entrez_ids: List of genes (Entrez IDs).
    - organism: Organism (default: 'hsapiens').
    - Ontology: Ontology and database to use for enrichment (default: ['GO:BP']).
    - evidences: Consider experimental evidence of terms.

    Retorna:
    - results_sorted: DataFrame with obtained terms ordered by p-value.
    """
    try:
        # Activate GProfiler instance and return dataframes.
        gp = GProfiler(return_dataframe=True)

        # Obtain dataframe with terms associates with list.
        results = gp.profile(
            organism=organism,                          # Species of study.
            query=entrez_ids,                           # EntrezID provided in input.
            user_threshold=0.05,                        # signigicant.
            sources=Ontology,                           # Source: GO:BP for example.
            no_evidences=not evidences                  # No use experimental evidence.
        )

        if results.empty:
            print("No terms found.")
            return pd.DataFrame()
        
        # Calculate some important extra columns.
        results.rename(columns={'precision': 'gene_ratio'}, inplace=True)
        results['qscore'] = -np.log10(results['p_value']) * results['gene_ratio']
        
        # Order by p-value.
        results_sorted = results.sort_values('p_value').reset_index(drop=True)
    except Exception as e:
        raise RuntimeError(f"Something went wrong: {e}")
    else:
        return results_sorted