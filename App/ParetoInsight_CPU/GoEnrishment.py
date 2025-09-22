######### Libraries #########
import numpy as np                   # Efficient Math Operations.
from gprofiler import GProfiler      # Web-server for enrichment analisys.                                    
import pandas as pd                  # Dataframe Managment.

######### AUX elements. #########

# Respective Taxonomy ID of species.
TAX_ID = {
    'human':     9606,
    'mouse':     10090,
    'fly':       7227,
    'zebrafish': 7955,
    'yeast':     559292,
    'athaliana': 3702,
    'schizosaccharomyces_pombe': 4896,
    'cow':       9913,
    'dog':       9615,
    'pig':       9823,
    'chicken':   9031,
    'rat':       10116,
    'c_elegans': 6239
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
    GoEnrichment (function): Perform GO enrichment analysis using entrezID.

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