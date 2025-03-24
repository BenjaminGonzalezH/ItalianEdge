if __name__ == "__main__":
    # Importaciones.
    import sys
    import os
    import time
    import numpy as np

    # Librerias propias.
    directory = os.path.dirname(__file__)
    sys.path.insert(0, os.path.abspath(os.path.join(directory, 'App/CoMOcG', 'src')))
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
    genes, num_genes, Matrix  = RD.ReadInputCSV_threads(file_1, n_workers = 6, solutions_id_colum=1)
    end_time = time.time()
    print(f"Tiempo de ejecución (lectura) : {end_time - start_time:.6f} segundos")

    ###### Matrices de conectividad.
    start_time = time.time()
    connec_sum = CM.connectivityMatrix_threads(Matrix,6)
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
    Ac.plot_html_heatmap(Prop_m,  directory + "/Results/File_1/Prop_matrix.html",
                        x_label="Gen",
                        y_label="Gen",
                        title="Similitud entre genes basada en asignaciones de grupos",
                        z_label="Proporción de coincidencia",
                        tooltip_format="Gen_ID_1: %{x}<br>Gen_ID_2: %{y}<br>Proporción: %{z:.2f}")

    ###### Cluster jerárquico.
    start_time = time.time()
    cons_cluster_1 = He.He_clustering_interactive(Dist_m, genes, 4, 
                                    save_path= directory + "/Results/File_1",
                                    dendrogram_file="Dendogram_file_1.html",
                                    method="single")
    Matrix = np.vstack([Matrix, cons_cluster_1])
    end_time = time.time()
    print(f"Tiempo de ejecución (Agrupamiento Jerarquico) : {end_time - start_time:.6f} segundos")

    ###### Comparación de composición de soluciones (JACCARD).
    start_time = time.time()
    Jaccard = JV.process_JaccardValues(Matrix, 6)
    Ac.plot_html_heatmap(Jaccard, save_filepath= directory + "/Results/File_1/JaccardS.html",
                        x_label='Solution',
                        y_label='Solution',
                        title='Similitud de Jaccard entre soluciones',
                        z_label="Jaccard",
                        tooltip_format="Solution_ID_1: %{x}<br>Solution_ID_2: %{y}<br>Jaccard: %{z:.2f}")
    Ac.save_matrix(Jaccard,  directory + "/Results/File_1/Jaccard_Matrix.csv")
    end_time = time.time()
    print(f"Tiempo de ejecución (Valores Jaccard) : {end_time - start_time:.6f} segundos")

    ###### Transformación de estrutura a conjunto de clusters por solución.
    start_time = time.time()
    SC_matrix = SCM.SolutionClusterMatrix_GeneID(Matrix, genes, 6)
    end_time = time.time()
    print(f"Tiempo de ejecución (Cambio de estructura) : {end_time - start_time:.6f} segundos")

    ###### Conversión a entrez ID.
    list_EIDs = GOe.convert_symbols_to_entrez_1(list(SC_matrix[0][0]), threads=6)

    ###### Enriquecimiento.
    Go_df = GOe.go_enrichment_entrez(list_EIDs)
    Go_df_1 = GOe.perform_go_enrichment(list_EIDs,convert_ids=False)
    Ac.save_dataframe(Go_df,directory + "/Results/File_1/Goedf.csv")
    Ac.save_dataframe(Go_df_1,directory + "/Results/File_1/Goedf_1.csv") 

    ###### Distancia de Wang.
    Wang_m = GOe.entrez_go_wang_similarity(list(SC_matrix[0][0]))