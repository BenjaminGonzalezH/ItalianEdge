######### Libraries #########
import concurrent.futures                           # Process Administration.
import numpy as np                                  # Efficient Math Operations.
import os                                           # OS callings.
from itertools import combinations                  # Eficient iterations.
import matplotlib                                   # Graph construction.
import pandas as pd
from ast import literal_eval


######### Configurations #########
#matplotlib.use('Agg')                               # Not GUI use, this is for conflicts between process.

######### Own Libraries #########
import JaccardValues as JV                          # Calculate Jaccard.
import SolutionComposition as SC                    # Calculate composition of functions.
import GoEnrischment as GOe                         # GO terms enrichment.
import Go_Plots as plots                            # Gene Ontology graph functions.
import Actions as Ac                                # Reiterative actions (heatmaps and saves).


######### Functions #########

def access_sets_only(df, groups_structure, organism="org.Hs.eg.db", ont="BP", convert_ids=True,
                          keytype="SYMBOL", directory=""):
    """
    Accede a los conjuntos y pares equivalentes SIN calcular promedios.
    
    Args:
        df: DataFrame con columnas ['Solution Pair', 'Equivalent Clusters']
        groups_structure: Lista de listas de conjuntos de IDs.
    
    Returns:
        Lista de tuplas con (grupo_i, grupo_j, conjunto_i, conjunto_j, pares_equivalentes)
    """
    
    # Parsear el DataFrame (sin modificar)
    def parse_entry(entry):
        if isinstance(entry, str):
            return literal_eval(entry.replace('np.float64', ''))
        return entry
    
    # Parsear el DataFrame
    df = df.copy()
    df['Solution Pair'] = df['Solution Pair'].apply(parse_entry)
    df['Equivalent Clusters'] = df['Equivalent Clusters'].apply(parse_entry)
    
    for _, row in df.iterrows():
        group_i, group_j = row['Solution Pair']
        equivalent_pairs = row['Equivalent Clusters']
        
        # Verificar índices de grupo
        if group_i >= len(groups_structure) or group_j >= len(groups_structure):
            continue
            
        # Obtener los conjuntos para estos grupos
        sets_i = groups_structure[group_i]
        sets_j = groups_structure[group_j]
        
        # Procesar cada par equivalente
        for elem_i, elem_j in equivalent_pairs:
            if elem_i >= len(sets_i) or elem_j >= len(sets_j):
                continue
                
            set_i = sets_i[elem_i]
            set_j = sets_j[elem_j]

            intersection = set_i.intersection(set_j)

            GO_DF = GOe.perform_go_enrichment(list(intersection), organism, ont, convert_ids, keytype)

            # Verificar si GO_DF está vacío o es None
            if GO_DF is None or GO_DF.empty:
                print(f"No se encontraron resultados de enriquecimiento GO para los clusters {elem_i},{elem_j} de los grupos {group_i},{group_j}")
            else:
                # Guardar y graficar solo si hay datos
                Ac.save_dataframe(GO_DF, directory + f"Enrichment_Solutions_{group_i},{group_j}_Clusters_{elem_i},{elem_j}.csv")
                plots.plot_gene_ratio_interactive(GO_DF, directory + f"GeneRatio_Solutions_{group_i},{group_j}_Clusters_{elem_i},{elem_j}.html")
                plots.plot_qscore_interactive(GO_DF, directory + f"Qscore_Solutions_{group_i},{group_j}_Clusters_{elem_i},{elem_j}.html")
                plots.plot_go_interaction_network_rpy2(list(intersection), save_path= directory + f"Network_Solutions_{group_i},{group_j}_Clusters_{elem_i},{elem_j}.png",
                                                    organism=organism, aspect=ont, convert_ids=convert_ids, keytype=keytype)
                plots.create_go_tree_rpy2(GO_DF, save_path= directory + f"Network_Solutions_{group_i},{group_j}_Clusters_{elem_i},{elem_j}.pdf")
