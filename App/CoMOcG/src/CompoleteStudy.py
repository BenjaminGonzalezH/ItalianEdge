######### Libraries #########
import concurrent.futures                           # Process Administration.
import numpy as np                                  # Efficient Math Operations.
import os                                           # OS callings.
from itertools import combinations                  # Eficient iterations.
import matplotlib                                   # Graph construction.

######### Configurations #########
#matplotlib.use('Agg')                               # Not GUI use, this is for conflicts between process.

######### Own Libraries #########
import JaccardValues as JV                          # Calculate Jaccard.
import SolutionComposition as SC                    # Calculate composition of functions.
import GoEnrischment as GOe                         # GO terms enrichment.
import Go_Plots as plots                            # Gene Ontology graph functions.
import Actions as Ac                                # Reiterative actions (heatmaps and saves).


######### Functions #########

"""
This block contains all main functions.
"""

def ensure_directory_exists(path: str) -> None:
    """
    ensure_directory_exists (functions):
    Check if filepath exists in your PC. If not, create one.

    Parameters:
    - path: Path to save the files. You need to ensure to write the path for your
      computer.
    """
    os.makedirs(path, exist_ok=True)

def analyze_solution_pair(
        pair: tuple[int,int], 
        SC_matrix: list[list[set]], 
        directory: str, 
        min_Jaccard_Value: int =0.2, 
        organism: str ="org.Hs.eg.db", 
        ont: str="BP", 
        convert_ids: bool=True):
    """
    analyze_solution_pair (function): Execute processes for do a GO terms analysis of a
    pair of functions.

    Parameters:
    - pair: Tuple with index of the solutions (pair).
    - SC_Matrix: Version of solutions that allocates gene symbol of their respective clusters.
    - directory: Path of your computer for save all the files generated.
    - min_Jaccard_Value: Minimal value of Jaccard Index (among pair of clusters between two solutions) for
      being study.
    - organism: Specie of study.
    - ont: Ontology from Gene Ontology.
    - convert_ids: Flag to declarate that gene symbols in SC_Matrix needs to be transform into Entrez ID.
    """
    try:
        # Extract sets.
        set1, set2 = SC_matrix[pair[0]], SC_matrix[pair[1]]
        
        # Calculate Jaccard Index among clusters and gene symbol
        # counts.
        jaccard_index = JV.Jaccar_similarityClusters(set1, set2)
        genes_count_matrix = SC.AmountGenes_Equals(set1, set2)
        
        # Generate path if it not exists.
        pair_id = f"{pair[0]}_{pair[1]}"
        ensure_directory_exists(directory)
        
        # Save Generated Matrix (and heatmap of jACCARD VALUES).
        Ac.plot_heatmap_matrix(jaccard_index, save_filepath=f"{directory}/Jaccard_E_D_{pair_id}.png",
                               x_label='Cluster', y_label='Cluster', title=f'Similitud Jaccard {pair_id}', show_flag=False)
        Ac.save_matrix_uncompresed(jaccard_index, save_filepath=f"{directory}/Jaccard_E_D_{pair_id}.csv")
        Ac.save_matrix_uncompresed(genes_count_matrix, save_filepath=f"{directory}/Genes_Count_E_D_{pair_id}.csv")

        # Main iteration.
        for i in range(len(jaccard_index)):
            for j in range(i + 1, len(jaccard_index)):
                # Check if Jaccard value in the coordenates is enough for study.
                if jaccard_index[i, j] >= min_Jaccard_Value:
                    # List from Set.
                    common_genes = set1[i].intersection(set2[j])
                    common_genes_list = list(common_genes)
                    
                    # Generate path if it not exists.
                    print(f"Calculating GO enrichment y Wang distance for {len(common_genes_list)} genes in clusters ({i}, {j})...")
                    result_id = f"{pair_id}_{i}_{j}"
                    sub_directory = f"{directory}/{i}_{j}"
                    ensure_directory_exists(sub_directory)

                    # GO enrichment
                    enrichment_results = GOe.perform_go_enrichment(common_genes_list, organism, ont, convert_ids)
                    Ac.save_dataframe(enrichment_results, f"{sub_directory}/enrichment_results_{result_id}.csv")
                    
                    # Wang Distance.
                    wang_matrix = GOe.calculate_wang_distance_matrix(common_genes_list, organism, ont, convert_ids)
                    Ac.save_dataframe(wang_matrix, f"{sub_directory}/wang_results_{result_id}.csv")

                    if not enrichment_results.empty and not wang_matrix.empty:
                        Ac.plot_heatmap_matrix(wang_matrix.to_numpy(), save_filepath=f"{sub_directory}/Wang_{result_id}.png",
                                               x_label='Genes', y_label='Genes', title=f'Wang Distance {result_id}', show_flag=False)
                        plots.plot_gene_ratio(enrichment_results, save_path=f"{sub_directory}/generatio_{result_id}.png", show_flag=False)
                        plots.plot_qscore(enrichment_results, save_path=f"{sub_directory}/qscore_{result_id}.png", show_flag=False)
                        plots.plot_go_interaction_network_rpy2(common_genes_list, similarity_threshold=0.7,
                                                               save_path=f"{sub_directory}/Network_terms_{result_id}.png")
                        plots.create_go_tree_rpy2(enrichment_results, save_path=f"{sub_directory}/tree_{result_id}.pdf")
                else:
                    print(f"Skipt analysis from clusters ({i}, {j}). Jaccard Index: {jaccard_index[i, j]} (min: {min_Jaccard_Value}).")

        return (pair, jaccard_index, genes_count_matrix, None)

    except Exception as e:
        print(f"Error in analyze_solution_pair for pair {pair}: {e}")
        return (pair, None, None, None)

def parallel_analysis(
        SC_matrix: list[list[set]], 
        directory: str, 
        min_Jaccard_Value: int =0.2, 
        organism: str ="org.Hs.eg.db", 
        ont: str="BP", 
        convert_ids: bool=True) -> None:
    """
    parallel_analysis (function): Execute processes for do a GO terms analysis of every pair of functions.

    Parameters:
    - SC_Matrix: Version of solutions that allocates gene symbol of their respective clusters.
    - directory: Path of your computer for save all the files generated.
    - min_Jaccard_Value: Minimal value of Jaccard Index (among pair of clusters between two solutions) for
      being study.
    - organism: Specie of study.
    - ont: Ontology from Gene Ontology.
    - convert_ids: Flag to declarate that gene symbols in SC_Matrix needs to be transform into Entrez ID.
    """
    # Create every pair of solutions (no inverted considerate).
    pairs = list(combinations(range(len(SC_matrix)), 2))
    
    # Procees managment.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(analyze_solution_pair, pair, SC_matrix, f"{directory}/{pair[0]}_{pair[1]}", min_Jaccard_Value, organism, ont, convert_ids): pair
            for pair in pairs
        }

        for future in concurrent.futures.as_completed(futures):
            pair = futures[future]
            try:
                result = future.result()
                print(f"Par {pair} procesado con éxito.")
            except Exception as e:
                print(f"Error al procesar el par {pair}: {e}")