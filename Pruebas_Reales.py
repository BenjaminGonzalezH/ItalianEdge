if __name__ == "__main__":
    # Importaciones.
    import os
    import time
    import numpy as np

    # Librerias propias.
    import App.ParetoInsight_CPU.ReadSolution as RD
    import App.ParetoInsight_CPU.ConsensusMatrix as CM
    import App.ParetoInsight_CPU.He_Clustering as He
    import App.ParetoInsight_CPU.SolutionClusterMatrix as SCM
    import App.ParetoInsight_CPU.JaccardValues as JV
    import App.ParetoInsight_CPU.RandValues as RV
    import App.ParetoInsight_CPU.MappingEntrez as ME
    import App.ParetoInsight_CPU.GoEnrishment as GOeP
    import ParetoInsight_CPU.WangIndex as WI
    import App.ParetoInsight_CPU.Actions as Ac
    import Graphs.Heatmaps as Heat
    import Graphs.Go_Plots as Gplot
    import Graphs.GoNetwork as Gnet
    import Graphs.Go_heiracialNetwork as GHnet

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
    Ac.save_matrix(Prop_m,  directory + "/Results/File_1/Prop_matrix.csv")
    Ac.save_matrix(Dist_m,  directory + "/Results/File_1/Dist_matrix.csv")
    Heat.plot_html_heatmap(Prop_m,  directory + "/Results/File_1/Prop_matrix.html",
                        x_label="Gen",
                        y_label="Gen",
                        title="Similitud entre genes basada en asignaciones de grupos",
                        z_label="Proporción de coincidencia",
                        tooltip_format="Gen_ID_1: %{x}<br>Gen_ID_2: %{y}<br>Proporción: %{z:.2f}")

    ###################################################################################################### Cluster jerárquico.
    start_time = time.time()
    cons_cluster_1 = He.He_clustering(Dist_m, genes, 4, 
                                    save_path= directory + "/Results/File_1",
                                    dendrogram_file="Dendogram_file_1.html",
                                    method="single")
    Matrix = np.vstack([Matrix, cons_cluster_1])
    end_time = time.time()
    print(f"Tiempo de ejecución (Agrupamiento Jerarquico) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Comparación de composición de soluciones (JACCARD).
    start_time = time.time()
    Jaccard = JV.JaccardIndexSolutions(Matrix, n_threads=8)
    end_time = time.time()
    print(f"Tiempo de ejecución (Valores Jaccard) : {end_time - start_time:.6f} segundos")
    Heat.plot_html_heatmap(Jaccard, save_filepath= directory + "/Results/File_1/JaccardS.html",
                        x_label='Solution',
                        y_label='Solution',
                        title='Similitud de Jaccard entre soluciones',
                        z_label="Jaccard",
                        tooltip_format="Solution_ID_1: %{x}<br>Solution_ID_2: %{y}<br>Jaccard: %{z:.2f}")
    Ac.save_matrix(Jaccard,  directory + "/Results/File_1/Jaccard_Matrix.csv")
    

    ###################################################################################################### Comparación de composición de clusters (JACCARD).
    entrez_gen = ME.ConvertToEntrezID(genes,organism_gp='hsapiens', taxID=9606)
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
    Ac.save_dataframe(df_equivalentes, directory + "/Results/File_1/Equivalentes.csv")

    ###################################################################################################### Obtener Entrez ID.
    start_time = time.time()
    EntrezID_P = ME.ConvertToEntrezID(list(SC_matrix[0][1]),organism_gp='hsapiens', taxID=9606)
    end_time = time.time()
    print(f"Tiempo de ejecución (EntrezID Python) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Go Enrichment.
    start_time = time.time()
    GO_DF_P = GOeP.GoEnrichment(EntrezID_P,organism='hsapiens')
    Ac.save_dataframe(GO_DF_P,directory + "/Results/File_1/Enrichment_Example_Python.csv")
    end_time = time.time()
    print(f"Tiempo de ejecución (Enriquecimiento biologico con entrezID Python) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Distancia de Wang.

    start_time = time.time()
    wang_matrix = WI.SimilarityIndexMatrix(genes,"goa_human",download_gaf=False)
    Heat.plot_html_heatmap(wang_matrix, save_filepath= directory + "/Results/File_1/Wang_genes.html",
                        x_label='Gen',
                        y_label='Gen',
                        title='Similitud de Wang entre genes',
                        z_label="Wang",
                        tooltip_format="Gen_ID_1: %{x}<br>Gen_ID_2: %{y}<br>Wang: %{z:.2f}")
    end_time = time.time()
    print(f"Tiempo de ejecución (Wang) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Toda la muestra (wang).
    start_time = time.time()
    wang_s = WI.Solution_Wang_index_similarity_Python(genes, wang_matrix, df_equivalentes, SC_matrix, num_threads=8)
    end_time = time.time()
    print(f"Tiempo de ejecución (Matriz) : {end_time - start_time:.6f} segundos")
    Heat.plot_dual_heatmap_two_colors(Jaccard, wang_s, directory + "/Results/File_1/Dual.html")

    ###################################################################################################### Dual Heatmap.

    EntrezID_P = ME.ConvertToEntrezID(list(SC_matrix[0][1]),organism_gp='athaliana', taxID=3702)
    GO_DF_P = GOeP.GoEnrichment(EntrezID_P, organism='athaliana')
    GtoT = WI.AnnotationFromEntrezIDs(EntrezID_P, Ontology=['GO:BP'], organism='athaliana')
    print(GtoT)
    term_pvalues = GO_DF_P.set_index("native")["p_value"].to_dict()
    print(term_pvalues)
    Gplot.plot_gene_ratio(GO_DF_P, directory + "/Results/File_1/GR.html")
    Gplot.plot_qscore(GO_DF_P, directory + "/Results/File_1/QS.html")
    Gnet.plot_go_interaction_network_html(GtoT, term_pvalues, 
                                          similarity_threshold=0.7,
                                          min_genes_per_term=5,
                                          max_node_size=30.0,
                                          save_path = directory + "/Results/File_1/Net.html")
    GHnet.plot_go_hierarchy_html(GtoT, 
                                 term_pvalues, 
                                 save_path=directory + "/Results/File_1/Tree.html")
    
    ###################################################################################################### Distancia de Wang.
