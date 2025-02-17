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
    genes, num_genes, Matrix  = RD.ReadInputCSV_threads(file_2, n_workers = 8, solutions_id_colum=0)
    end_time = time.time()
    print(f"Tiempo de ejecución (lectura) : {end_time - start_time:.6f} segundos")

    ###### Matrices de conectividad.
    start_time = time.time()
    connec_sum = CM.connectivityMatrix_threads(Matrix,8)
    connec_sum = CM.sum_connectivity_matrices(connec_sum)
    Ac.save_matrix(connec_sum.toarray(), directory + "/Results/File_2/Connectivity_Matrix.csv")
    end_time = time.time()
    print(f"Tiempo de ejecución (conectividad) : {end_time - start_time:.6f} segundos")

    ###### Matriz de proporción y distancia.
    start_time = time.time()
    Prop_m, Dist_m = PM.ProportionsMatrix(connec_sum)
    Ac.save_matrix(Prop_m,  directory + "/Results/File_2/Prop_matrix.csv")
    Ac.save_matrix(Dist_m,  directory + "/Results/File_2/Dist_matrix.csv")
    end_time = time.time()
    print(f"Tiempo de ejecución (Proporcion y distancia) : {end_time - start_time:.6f} segundos")
    Ac.plot_html_heatmap(Prop_m,  directory + "/Results/File_2/Prop_matrix.html",
                        x_label="Solution",
                        y_label="Solution",
                        title="Similitud por prescencia de genes en clusters")
    Ac.plot_html_heatmap(Dist_m,  directory + "/Results/File_2/Dist_matrix.html",
                        x_label="Solution",
                        y_label="Solution",
                        title="Distancia por prescencia de genes en clusters",
                        color='plasma')

    ###### Cluster jerárquico.
    start_time = time.time()
    cons_cluster = He.He_clustering(Dist_m, genes, 4, 
                                    save_path= directory + "/Results/File_2",
                                    dendrogram_file="Dendogram_file_2.png", show_flag=False)
    end_time = time.time()
    print(f"Tiempo de ejecución (Agrupamiento Jerarquico) : {end_time - start_time:.6f} segundos")

    ###### Transformación de estrutura a conjunto de clusters por solución.
    start_time = time.time()
    SC_matrix = SCM.SolutionClusterMatrix_GeneID(Matrix, genes, 8)
    end_time = time.time()
    print(f"Tiempo de ejecución (Cambio de estructura) : {end_time - start_time:.6f} segundos")

    ###### Comparación de composición de soluciones (JACCARD).
    start_time = time.time()
    Jaccard = JV.process_JaccardValues(Matrix, 8)
    Ac.plot_heatmap_matrix(Jaccard, save_filepath= directory + "/Results/File_2/JaccardS.png",
                        x_label='Solution',
                        y_label='Solution',
                        title='Similitud de Jaccard entre soluciones',
                        color='cividis',
                        show_flag=False)
    Ac.plot_heatmap_matrix(1-Jaccard, save_filepath= directory + "/Results/File_2/JaccardD.png",
                        x_label='Solution',
                        y_label='Solution',
                        title='Distancia de Jaccard entre soluciones',
                        color='inferno',
                        show_flag=False)
    Ac.save_matrix(1-Jaccard,  directory + "/Results/File_2/Jaccard_Matrix.csv")
    end_time = time.time()
    print(f"Tiempo de ejecución (Valores Jaccard) : {end_time - start_time:.6f} segundos")

    ###### Comparación de composición de soluciones (LISTAS).
    start_time = time.time()
    Coposition = SC.process_proportion_genessolution(Matrix, 8)
    end_time = time.time()
    Ac.plot_heatmap_matrix(Coposition, save_filepath= directory + "/Results/File_2/CopositionS.png",
                        x_label='Solution',
                        y_label='Solution',
                        title='Similitud de Composición entre soluciones',
                        color='cividis',
                        show_flag=False)
    Ac.plot_heatmap_matrix(1-Coposition, save_filepath= directory + "/Results/File_2/CopositionD.png",
                        x_label='Solution',
                        y_label='Solution',
                        title='Distancia de Composición entre soluciones',
                        color='inferno',
                        show_flag=False)
    Ac.save_matrix(1-Coposition,  directory + "/Results/File_2/Coposition_Matrix.csv")
    end_time = time.time()
    print(f"Tiempo de ejecución (Composición de Listas) : {end_time - start_time:.6f} segundos")

    ###### Estudio final.
    start_time = time.time()
    CS.parallel_analysis(SC_matrix[0:5], directory + "/Results/File_2_Global", min_Jaccard_Value=0.2, convert_ids=True)
    end_time = time.time()
    print(f"Tiempo de ejecución (Análisis completo) : {end_time - start_time:.6f} segundos")
