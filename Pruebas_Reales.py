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
    import App.CoMOcG.MappingEntrez as ME
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
    Matrix, genes  = RD.ReadSolutionsFile(file_3,"csv")
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
    Jaccard = JV.JaccardIndexSolutions(Matrix, n_threads=8)
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
    entrez_gen = GOe.convert_symbols_to_entrez(genes,organism="org.At.tair.db", keytype="TAIR")
    SC_matrix = SCM.SolutionClusterMatrix(Matrix, entrez_gen, 8)
    start_time = time.time()
    Jaccard_c = JV.JaccardIndexClusters(SC_matrix[0], SC_matrix[1])
    end_time = time.time()
    print(f"Tiempo de ejecución (Comparar clusters - Jaccard) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Comparación de composición de clusters (RAND).
    start_time = time.time()
    Rand = RV.RandIndexSolutions(Matrix, 8)
    end_time = time.time()
    print(f"Tiempo de ejecución (Valores RI) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Reformateo de solución (Solucion-Cluster).  
    start_time = time.time()
    SC_matrix = SCM.SolutionClusterMatrix(Matrix, genes, 8)
    end_time = time.time()
    print(f"Tiempo de ejecución (SCM) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Obtener clusters equivalentes.  
    start_time = time.time()
    df_equivalentes = JV.FindEquivalentClusters(SC_matrix)
    end_time = time.time()
    print(f"Tiempo de ejecución (grupos equivalentes) : {end_time - start_time:.6f} segundos")
    Ac.save_dataframe(df_equivalentes, directory + "/Results/File_3/Equivalentes.csv")

    ###################################################################################################### Obtener Entrez ID.
    start_time = time.time()
    EntrezID_P = ME.ConvertToEntrezID(list(SC_matrix[0][1]), organism_gp='athaliana', taxID= 3702)
    end_time = time.time()
    print(f"Tiempo de ejecución (EntrezID Python) : {end_time - start_time:.6f} segundos")

    sc_COPY = {str(x) for x in SC_matrix[0][1]}
    start_time = time.time()
    EntrezID_r = GOe.convert_symbols_to_entrez(list(sc_COPY),organism="org.At.tair.db",keytype="TAIR")
    end_time = time.time()
    print(f"Tiempo de ejecución (EntrezID r) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Go Enrichment.
    start_time = time.time()
    GO_DF_P = GOeP.GoEnrichment(EntrezID_P, organism='athaliana')
    Ac.save_dataframe(GO_DF_P,directory + "/Results/File_3/Enrichment_Example_Python.csv")
    end_time = time.time()
    print(f"Tiempo de ejecución (Enriquecimiento biologico con entrezID Python) : {end_time - start_time:.6f} segundos") 
    
    start_time = time.time()
    Go_df_now = GOe.perform_go_enrichment(EntrezID_r,organism="org.At.tair.db",keytype="TAIR", convert_ids=False)
    Ac.save_dataframe(Go_df_now,directory + "/Results/File_3/Enrichment_Example_R.csv")
    end_time = time.time()
    print(f"Tiempo de ejecución (Enriquecimiento biologico con entrezID R) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Distancia de Wang.
    start_time = time.time()
    Wang_df = GOe.calculate_wang_distance_matrix(list(sc_COPY), organism="org.At.tair.db", convert_ids=True, keytype="TAIR")
    end_time = time.time()
    print(f"Tiempo de ejecución (Wang R) : {end_time - start_time:.6f} segundos")