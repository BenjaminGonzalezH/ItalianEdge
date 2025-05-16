if __name__ == "__main__":
    # Importaciones.
    import os
    import time
    import numpy as np

    # Librerias propias.
    import App.CoMOcG.ReadSolution as RD
    import App.CoMOcG.ConsensusMatrix as CM
    import App.CoMOcG.He_Clustering as He
    import App.CoMOcG.SolutionClusterMatrix as SCM
    import App.CoMOcG.JaccardValues as JV
    import App.CoMOcG.RandValues as RV
    import App.CoMOcG.GoEnrischment as GOe
    import App.CoMOcG.Actions as Ac
    import App.CoMOcG.CompoleteStudy as CS
    import App.CoMOcG.GoEnrishmentPy as GOeP
    import App.CoMOcG.Go_Plots as Gplot

    # Obtain actual directory.
    directory = os.path.dirname(__file__)

    # Obtain test files.
    file_1 = directory + r"\test_files_reals\archivo_prueba_1_116_500.csv"
    file_2 = directory + r"\test_files_reals\archivo_prueba_2_116_3444.csv"
    file_3 = directory + r"\test_files_reals\archivo_prueba_3_25_133.csv"

    ###################################################################################################### Lectura de archivo.
    start_time = time.time()
    Matrix, genes  = RD.ReadSolutionsFile(file_1,"csv")
    end_time = time.time()
    print(f"Tiempo de ejecución (lectura) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Matriz de consenso.
    start_time = time.time()
    Prop_m, Dist_m = CM.ConsensusMatrix(Matrix)
    end_time = time.time()
    print(f"Tiempo de ejecución (consenso) : {end_time - start_time:.6f} segundos")
    Ac.save_matrix(Prop_m,  directory + "/Results/File_3/Prop_matrix.csv")
    Ac.save_matrix(Dist_m,  directory + "/Results/File_3/Dist_matrix.csv")
    Ac.plot_html_heatmap(Prop_m,  directory + "/Results/File_3/Prop_matrix.html",
                        x_label="Gen",
                        y_label="Gen",
                        title="Similitud entre genes basada en asignaciones de grupos",
                        z_label="Proporción de coincidencia",
                        tooltip_format="Gen_ID_1: %{x}<br>Gen_ID_2: %{y}<br>Proporción: %{z:.2f}")

    ###################################################################################################### Cluster jerárquico.
    start_time = time.time()
    cons_cluster_1 = He.He_clustering(Dist_m, genes, 4, 
                                    save_path= directory + "/Results/File_3",
                                    dendrogram_file="Dendogram_file_3.html",
                                    method="single")
    Matrix = np.vstack([Matrix, cons_cluster_1])
    end_time = time.time()
    print(f"Tiempo de ejecución (Agrupamiento Jerarquico) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Comparación de composición de soluciones (JACCARD).
    start_time = time.time()
    Jaccard = JV.JaccardIndexSolutions(Matrix)
    end_time = time.time()
    print(f"Tiempo de ejecución (Valores Jaccard) : {end_time - start_time:.6f} segundos")
    Ac.plot_html_heatmap(Jaccard, save_filepath= directory + "/Results/File_3/JaccardS.html",
                        x_label='Solution',
                        y_label='Solution',
                        title='Similitud de Jaccard entre soluciones',
                        z_label="Jaccard",
                        tooltip_format="Solution_ID_1: %{x}<br>Solution_ID_2: %{y}<br>Jaccard: %{z:.2f}")
    Ac.save_matrix(Jaccard,  directory + "/Results/File_3/Jaccard_Matrix.csv")
    

    ###################################################################################################### Comparación de composición de clusters (JACCARD).
    entrez_gen = GOe.convert_symbols_to_entrez(genes)
    SC_matrix = SCM.SolutionClusterMatrix(Matrix, entrez_gen, 6)
    start_time = time.time()
    Jaccard_c = JV.JaccardIndexClusters(SC_matrix[0], SC_matrix[1])
    end_time = time.time()
    print(f"Tiempo de ejecución (Comparar clusters - Jaccard) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Comparación de composición de clusters (RAND).
    start_time = time.time()
    Rand = RV.RandIndexSolutions(Matrix)
    end_time = time.time()
    print(f"Tiempo de ejecución (Valores RI) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Reformateo de solución (Solucion-Cluster).  
    start_time = time.time()
    SC_matrix = SCM.SolutionClusterMatrix(Matrix, genes)
    end_time = time.time()
    print(f"Tiempo de ejecución (SCM) : {end_time - start_time:.6f} segundos") 

    start_time = time.time()
    SC_matrix_a = SCM.process_solution_vectorized(Matrix, genes)
    end_time = time.time()
    print(f"Tiempo de ejecución (SCM) : {end_time - start_time:.6f} segundos")  
    
    
    start_time = time.time()
    SC_matrix = SCM.SolutionClusterMatrix(Matrix, entrez_gen, 6)
    end_time = time.time()
    print(f"Tiempo de ejecución (Cambio de estructura) : {end_time - start_time:.6f} segundos")

    ###### Distancia de Wang en la totalidad de genes.
    start_time = time.time()
    Wang_df = GOe.calculate_wang_distance_matrix(genes, organism="org.At.tair.db", convert_ids=True, keytype="TAIR")
    end_time = time.time()
    print(f"Tiempo de ejecución (W 2) : {end_time - start_time:.6f} segundos")
    Ac.save_dataframe(Wang_df, directory + "/Results/File_3/Wang.csv")

    ###### Búsqueda de clusters equivalentes entre soluciones.
    start_time = time.time()
    df_equivalentes = JV.find_equivalent_clusters(SC_matrix)
    end_time = time.time()
    print(f"Tiempo de ejecución (grupos equivalentes) : {end_time - start_time:.6f} segundos")
    Ac.save_dataframe(df_equivalentes, directory + "/Results/File_3/Equivalentes.csv")

    ###### Similitud de wang entre soluciones.
    Genes_wang = list(Wang_df.columns.values)
    Wang_Matrix = Wang_df.to_numpy()
    start_time = time.time()
    Sim_Wang = GOe.Solution_Wang_index_similarity_Python(Genes_wang, Wang_Matrix, df_equivalentes, SC_matrix, num_threads=6)
    end_time = time.time()
    print(f"Tiempo de ejecución (grupos similares en wang) : {end_time - start_time:.6f} segundos") 
    Ac.plot_html_heatmap(Sim_Wang,  directory + "/Results/File_3/Prop_matrix_P_2.html",
                        x_label="Solution",
                        y_label="Solution",
                        title="Similitud entre soluciones usando WANG",
                        z_label="Wang",
                        tooltip_format="Gen_ID_1: %{x}<br>Gen_ID_2: %{y}<br>Wang: %{z:.2f}")


    ###### Python Propolsals.
    SC_matrix_1 = SCM.SolutionClusterMatrix(Matrix, genes, 6)
    start_time = time.time()
    EntrezID_Pg = GOeP.convert_symbols_to_entrez_Python(genes, organism_gp='athaliana', taxID= 3702)
    EntrezID_P = GOeP.convert_symbols_to_entrez_Python(list(SC_matrix_1[0][1]), organism_gp='athaliana', taxID= 3702)
    end_time = time.time()
    print(f"Tiempo de ejecución (Conversion EntrezID Python) : {end_time - start_time:.6f} segundos")

    start_time = time.time()
    GO_DF_P = GOeP.go_enrichment_entrez_Python(EntrezID_P, organism='athaliana')
    Ac.save_dataframe(GO_DF_P,directory + "/Results/File_3/Enrichment_Example_Python.csv")
    end_time = time.time()
    print(f"Tiempo de ejecución (Enriquecimiento biologico con entrezID) : {end_time - start_time:.6f} segundos") 


    ###### Imprimir superposición de matrices.
    start_time = time.time()
    Ac.plot_dual_heatmap_two_colors(np.triu(Jaccard), np.tril(Sim_Wang), save_filepath= directory + "/Results/File_3/DUAL_W_J.html")
    end_time = time.time()
    print(f"Tiempo de ejecución (matrices superpuestas) : {end_time - start_time:.6f} segundos") 

    ###### Gráfico entre soliciones.
    start_time = time.time()
    Go_df_now = GOe.perform_go_enrichment(list(SC_matrix_1[0][1]),organism="org.At.tair.db",keytype="TAIR")
    end_time = time.time()
    print(f"Tiempo de ejecución (grupos similares en wang) : {end_time - start_time:.6f} segundos") 
    Ac.save_dataframe(Go_df_now, directory + "/Results/File_3/Enrichment_Example_R.csv")
    Gplot.plot_gene_ratio(Go_df_now, directory + "/Results/File_3/Gene_Ratio_Example.html")
    Gplot.plot_qscore(Go_df_now, directory + "/Results/File_3/Qscore_Example.html")
    Gplot.plot_go_interaction_network_rpy2(list(SC_matrix_1[0][1]),organism="org.At.tair.db", save_path=directory + "/Results/File_3/Network_Example.png",
                                           keytype="TAIR", width=1920, height=1080)
    Gplot.create_go_tree_rpy2(Go_df_now, save_path=directory + "/Results/File_3/Tree.pdf")
    end_time = time.time()
    print(f"Tiempo de ejecución (Go plots) : {end_time - start_time:.6f} segundos") 

    # Estudio completo.
    #CS.Complete_Study(df_equivalentes,SC_matrix, convert_ids=False, directory= directory + "/Results/File_3/Global/")

