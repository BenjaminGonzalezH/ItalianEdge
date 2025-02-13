if __name__ == "__main__":
    # Importaciones.
    import sys
    import os
    import time
    import numpy as np

    # Librerias propias.
    directory = os.path.dirname(__file__)
    sys.path.insert(0, os.path.abspath(os.path.join(directory, 'App\CoMOcG', 'src')))
    import ReadSolution as RD
    import ConnectivityMatrix as CM
    import ProportionMatrix as PM
    import He_Clustering as He
    import SolutionClusterMatrix as SCM
    import JaccardValues as JV
    import SolutionComposition as SC
    import GoEnrischment as GOe
    import Go_Plots as plots
    import Actions as Ac
    import CompoleteStudy as CS
    from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
    import logging

    rpy2_logger.setLevel(logging.ERROR)  # Suprimir mensajes de R

    # Obtain actual directory.
    directory = os.path.dirname(__file__)

    # Obtain test files.
    file_1 = directory + r"\test_files_reals\archivo_prueba_1_116_500.csv"
    file_2 = directory + r"\test_files_reals\archivo_prueba_2_116_3444.csv"
    file_3 = directory + r"\test_files_reals\archivo_prueba_3_25_133.csv"

    ###### Lectura de archivo.
    start_time = time.time()
    genes, num_genes, Matrix  = RD.ReadInputCSV_threads(file_2, n_workers = 8, solutions_id_colum=0)
    end_time = time.time()
    print(f"Tiempo de ejecución (lectura) : {end_time - start_time:.6f} segundos")

    ###### Matrices de conectividad.
    start_time = time.time()
    connec_sum = CM.connectivityMatrix_threads(Matrix,8)
    connec_sum = CM.sum_connectivity_matrices(connec_sum)
    Ac.save_matrix(connec_sum.toarray(), directory + "/Results/File_1/Connectivity_Matrix.csv")
    end_time = time.time()
    print(f"Tiempo de ejecución (conectividad) : {end_time - start_time:.6f} segundos")

    ###### Matriz de proporción y distancia.
    start_time = time.time()
    Prop_m, Dist_m = PM.ProportionsMatrix(connec_sum)
    Ac.save_matrix(Prop_m,  directory + "/Results/File_1/Prop_matrix.csv")
    Ac.save_matrix(Dist_m,  directory + "/Results/File_1/Dist_matrix.csv")
    end_time = time.time()
    print(f"Tiempo de ejecución (Proporcion y distancia) : {end_time - start_time:.6f} segundos")
    Ac.plot_heatmap_matrix(Prop_m,  directory + "/Results/File_1/Prop_matrix.png",
                        x_label="Solution",
                        y_label="Solution",
                        title="Similitud por prescencia de genes en clusters",
                        show_flag=False)
    Ac.plot_heatmap_matrix(Dist_m,  directory + "/Results/File_1/Dist_matrix.png",
                        x_label="Solution",
                        y_label="Solution",
                        title="Distancia por prescencia de genes en clusters",
                        color='plasma',
                        show_flag=False)

    ###### Cluster jerárquico.
    start_time = time.time()
    cons_cluster = He.He_clustering(Dist_m, genes, 4, 
                                    save_path= directory + "/Results/File_1",
                                    dendrogram_file="Dendogram_file_1.png", show_flag=False)
    end_time = time.time()
    print(f"Tiempo de ejecución (Agrupamiento Jerarquico) : {end_time - start_time:.6f} segundos")

    ###### Transformación de estrutura a conjunto de clusters por solución.
    start_time = time.time()
    SC_matrix = SCM.SolutionClusterMatrix_GeneID(Matrix, genes, 8)
    end_time = time.time()
    print(f"Tiempo de ejecución (Cambio de estructura) : {end_time - start_time:.6f} segundos")

    start_time = time.time()
    # Configuración de tus datos
    CS.parallel_analysis(SC_matrix[0:10], directory + "/Results/File_1_Global", min_gene_count=40)
    end_time = time.time()
    print(f"Estudio global : {end_time - start_time:.6f} segundos")

###### Comparación númerica de soluciones (Jaccard y Composición)
"""start_time = time.time()
Jaccard = JV.process_JaccardValues(Matrix, 8)
end_time = time.time()
print(f"Tiempo de ejecución (Valores Jaccard) : {end_time - start_time:.6f} segundos")
Ac.plot_heatmap_matrix(Jaccard, save_filepath= directory + "/Results/File_1/JaccardS.png",
                       x_label='Solution',
                       y_label='Solution',
                       title='Similitud de Jaccard entre soluciones',
                       color='cividis',
                       show_flag=False)
Ac.plot_heatmap_matrix(1-Jaccard, save_filepath= directory + "/Results/File_1/JaccardD.png",
                       x_label='Solution',
                       y_label='Solution',
                       title='Distancia de Jaccard entre soluciones',
                       color='inferno',
                       show_flag=False)
Ac.save_matrix(1-Jaccard,  directory + "/Results/File_1/Jaccard_Matrix.csv")


start_time = time.time()
Coposition = SC.process_proportion_genessolution(Matrix, 8)
end_time = time.time()
print(f"Tiempo de ejecución (Composición de composición) : {end_time - start_time:.6f} segundos")
Ac.plot_heatmap_matrix(Coposition, save_filepath= directory + "/Results/File_1/CopositionS.png",
                       x_label='Solution',
                       y_label='Solution',
                       title='Similitud de Composición entre soluciones',
                       color='cividis',
                       show_flag=False)
Ac.plot_heatmap_matrix(1-Coposition, save_filepath= directory + "/Results/File_1/CopositionD.png",
                       x_label='Solution',
                       y_label='Solution',
                       title='Distancia de Composición entre soluciones',
                       color='inferno',
                       show_flag=False)
Ac.save_matrix(1-Coposition,  directory + "/Results/File_1/Coposition_Matrix.csv")
"""

###### Comparación entre 2 soluciones arbitrarias (Jaccard y Cantidad de genes).
#start_time = time.time()
#Jaccard_E = JV.Jaccar_similarityClusters(SC_matrix[0], SC_matrix[1])
#end_time = time.time()
#print(f"Tiempo de ejecución (Comparar dos soluciones - Jaccard) : {end_time - start_time:.6f} segundos")

#start_time = time.time()
#Coposition_E = SC.AmountGenes_Equals(SC_matrix[0], SC_matrix[1])
#end_time = time.time()
#print(f"Tiempo de ejecución (Comparar dos soluciones cant_genes) : {end_time - start_time:.6f} segundos")
#Ac.plot_heatmap_matrix(1-Jaccard_E, save_filepath= directory + "/Results/File_1/Jaccard_E_D.png",
#                       x_label='Solution',
#                       y_label='Solution',
#                       title='Distancia de Composición entre soluciones',
#                       color='inferno',
#                       show_flag=False)
#Ac.plot_heatmap_matrix(Coposition_E, save_filepath= directory + "/Results/File_1/Coposition_E_S.png",
#                       x_label='Solution',
#                       y_label='Solution',
#                       title='Distancia de Composición entre soluciones',
#                       color='cividis',
#                       show_flag=False)

###### Go Enrichment.
#start_time = time.time()
#df_enrichment = GOe.perform_go_enrichment(list(SC_matrix[0][0]))
#end_time = time.time()
#print(f"Tiempo de ejecución (Enriquecimiento con términos) : {end_time - start_time:.6f} segundos")
#Ac.save_dataframe(df_enrichment, directory + "/Results/File_1/enrichment_results.csv" )
#print("La cantidad de genes analizada es de {}".format(len(list(SC_matrix[0][0]))))

###### Transformación a terminos.
#start_time = time.time()
#Genes_ID = GOe.convert_symbols_to_entrez(list(SC_matrix[0][0]))
#end_time = time.time()
#print(f"Tiempo de ejecución (Transformación de Terminos a Gene_ID) : {end_time - start_time:.6f} segundos")

###### Distancia de Wang Entre genes.
#start_time = time.time()
#Wang = GOe.calculate_wang_distance_matrix(list(SC_matrix[0][0]))
#end_time = time.time()
#print(f"Tiempo de ejecución (Calculo de distancia de Wang) : {end_time - start_time:.6f} segundos")
#Ac.plot_heatmap_matrix(Wang.to_numpy(), save_filepath= directory + "/Results/File_1/Wang.png",
#                       x_label='genes',
#                       y_label='genes',
#                       title='Distancia de Wang entre grupo de genes analizado',
#                       show_flag=False)
#Ac.save_dataframe(Wang, directory + "/Results/File_1/wang_results.csv" )

###### Plots GO.
#plots.plot_gene_ratio(df_enrichment, save_path= directory + "/Results/File_1/generatio.png", show_flag= False)
#plots.plot_qscore(df_enrichment, save_path= directory + "/Results/File_1/qscore.png", show_flag= False)
#plots.plot_go_interaction_network_rpy2(list(SC_matrix[0][0]), 
#                                       similarity_threshold=0.7, 
#                                       save_path= directory + "/Results/File_1/Network_terms.png")
#plots.create_go_tree_rpy2(df_enrichment,save_path= directory + "/Results/File_1/tree.pdf")

