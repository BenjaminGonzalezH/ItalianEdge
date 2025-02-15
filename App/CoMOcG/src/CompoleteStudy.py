# Libraries
import JaccardValues as JV
import SolutionComposition as SC
import GoEnrischment as GOe
import Go_Plots as plots
import Actions as Ac

import concurrent.futures
import numpy as np
import os
from itertools import combinations
import matplotlib
import concurrent.futures
from itertools import combinations

matplotlib.use('Agg')

# Function.
def ensure_directory_exists(path):
    """Crea el directorio si no existe."""
    os.makedirs(path, exist_ok=True)

def analyze_solution_pair(pair, SC_matrix, directory, min_Jaccard_Value=0.2, organism="org.Hs.eg.db", ont="BP", convert_ids=True):
    """Analiza un par de soluciones y filtra por índice de Jaccard para realizar GO enrichment y calcular distancia de Wang."""
    try:
        set1, set2 = SC_matrix[pair[0]], SC_matrix[pair[1]]
        
        jaccard_index = JV.Jaccar_similarityClusters(set1, set2)
        genes_count_matrix = SC.AmountGenes_Equals(set1, set2)
        
        pair_id = f"{pair[0]}_{pair[1]}"
        ensure_directory_exists(directory)
        
        # Guardar matrices de Jaccard y conteo de genes
        Ac.plot_heatmap_matrix(jaccard_index, save_filepath=f"{directory}/Jaccard_E_D_{pair_id}.png",
                               x_label='Cluster', y_label='Cluster', title=f'Similitud Jaccard {pair_id}', show_flag=False)
        Ac.save_matrix_uncompresed(jaccard_index, save_filepath=f"{directory}/Jaccard_E_D_{pair_id}.csv")
        Ac.save_matrix_uncompresed(genes_count_matrix, save_filepath=f"{directory}/Genes_Count_E_D_{pair_id}.csv")

        for i in range(len(jaccard_index)):
            for j in range(i + 1, len(jaccard_index)):
                if jaccard_index[i, j] >= min_Jaccard_Value:  # Usar el índice de Jaccard como filtro
                    common_genes = set1[i].intersection(set2[j])
                    common_genes_list = list(common_genes)

                    if not common_genes_list:
                        print(f"No se encontraron genes comunes en posición ({i}, {j}).")
                        continue
                    
                    print(f"Procesando GO enrichment y Wang distance para {len(common_genes_list)} genes en posición ({i}, {j})...")
                    result_id = f"{pair_id}_{i}_{j}"
                    sub_directory = f"{directory}/{i}_{j}"
                    ensure_directory_exists(sub_directory)

                    # Calcular GO enrichment
                    enrichment_results = GOe.perform_go_enrichment(common_genes_list, organism, ont, convert_ids)
                    Ac.save_dataframe(enrichment_results, f"{sub_directory}/enrichment_results_{result_id}.csv")
                    
                    # Calcular distancia de Wang
                    wang_matrix = GOe.calculate_wang_distance_matrix(common_genes_list, organism, ont, convert_ids)
                    Ac.save_dataframe(wang_matrix, f"{sub_directory}/wang_results_{result_id}.csv")

                    if not enrichment_results.empty and not wang_matrix.empty:
                        Ac.plot_heatmap_matrix(wang_matrix.to_numpy(), save_filepath=f"{sub_directory}/Wang_{result_id}.png",
                                               x_label='Genes', y_label='Genes', title=f'Distancia de Wang {result_id}', show_flag=False)
                        plots.plot_gene_ratio(enrichment_results, save_path=f"{sub_directory}/generatio_{result_id}.png", show_flag=False)
                        plots.plot_qscore(enrichment_results, save_path=f"{sub_directory}/qscore_{result_id}.png", show_flag=False)
                        plots.plot_go_interaction_network_rpy2(common_genes_list, similarity_threshold=0.7,
                                                               save_path=f"{sub_directory}/Network_terms_{result_id}.png")
                        plots.create_go_tree_rpy2(enrichment_results, save_path=f"{sub_directory}/tree_{result_id}.pdf")
                else:
                    print(f"Saltando análisis para posición ({i}, {j}). Índice de Jaccard: {jaccard_index[i, j]} (umbral: {min_Jaccard_Value}).")

        return (pair, jaccard_index, genes_count_matrix, None)

    except Exception as e:
        print(f"Error en analyze_solution_pair para el par {pair}: {e}")
        return (pair, None, None, None)

def parallel_analysis(SC_matrix, directory, min_Jaccard_Value=0.2, organism="org.Hs.eg.db", ont="BP", convert_ids=True):
    """Ejecuta el análisis en paralelo para todos los pares de soluciones utilizando ProcessPoolExecutor."""
    pairs = list(combinations(range(len(SC_matrix)), 2))
    
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
