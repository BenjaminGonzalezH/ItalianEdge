if __name__ == "__main__":
    # Importaciones.
    import os
    import time
    import numpy as np

    # Librerias propias.
    import App.ParetoInsight_CPU.ReadSolution as RD
    import App.ParetoInsight_CPU.ConsensusMatrix as CM
    import App.ParetoInsight_CPU.He_Clustering as He
    import App.ParetoInsight_CPU.EssemblingClustering as EssCl
    import App.ParetoInsight_CPU.SolutionClusterMatrix as SCM
    import App.ParetoInsight_CPU.JaccardValues as JV
    import App.ParetoInsight_CPU.RandValues as RV
    import App.ParetoInsight_CPU.MappingEntrez as ME
    import App.ParetoInsight_CPU.GoEnrishment as GOeP
    import App.ParetoInsight_CPU.WangIndex as WI
    import App.ParetoInsight_CPU.Actions as Ac
    import App.ParetoInsight_CPU.SimilarityThreshold as ST
    import App.Graphs.Heatmaps as Heat
    import App.Graphs.Go_Plots as Gplot
    import App.Graphs.GoNetwork as Gnet
    import App.Graphs.Go_heiracialNetwork as GHnet
    import App.Graphs.Raincloud as RC
    import App.Graphs.GoChordDiagram as GCD

    # Obtain actual directory.
    directory = os.path.dirname(__file__)

    # Obtain test files.
    file_1 = directory + r"\Test_Files\archivo_prueba_1_116_500.csv"
    file_2 = directory + r"\Test_Files\archivo_prueba_2_116_3444.csv"
    file_3 = directory + r"\Test_Files\archivo_prueba_3_25_133.csv"

    ###################################################################################################### Lectura de archivo.
    start_time = time.time()
    Matrix, genes  = RD.read_solutions_file(file_1)
    end_time = time.time()
    print(f"Tiempo de ejecución (lectura) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Matriz de consenso.
    start_time = time.time()
    Prop_m, Dist_m = CM.consensus_matrix(Matrix)
    end_time = time.time()
    print(f"Tiempo de ejecución (consenso) : {end_time - start_time:.6f} segundos")
    options = Ac.MatrixSaveOptions(mode=Ac.MatrixSaveMode.TEXT_CSV, verbose=True, delimiter=";")
    Ac.save_matrix(Prop_m,  directory + "/Results/File_1/Prop_matrix.csv",options)
    Ac.save_matrix(Dist_m,  directory + "/Results/File_1/Dist_matrix.csv",options)
    Heat.plot_html_heatmap(Prop_m,  
                           directory + "/Results/File_1/Prop_matrix.html",
                            x_label="Gen",
                            y_label="Gen",
                            title="Similitud entre genes basada en asignaciones de grupos",
                            z_label="Proporción de coincidencia",
                            tooltip_format="Gen_ID_1: %{x}<br>Gen_ID_2: %{y}<br>Proporción: %{z:.2f}")

    ###################################################################################################### Cluster jerárquico (Co-Ocurrencia + Agr. Jerarquico).
    start_time = time.time()
    cons_cluster_1 = He.he_clustering(Dist_m, 
                                        genes,
                                        save_html_to= directory + "/Results/File_1/Dendogram_file_1.html")
    Matrix = np.vstack([Matrix, cons_cluster_1])
    end_time = time.time()
    print(f"Tiempo de ejecución (Agrupamiento Jerarquico) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Essembling Clustering.
    start_time = time.time()
    CSPAoptions = EssCl.CSPAOptions(n_clusters=4,assign_labels="kmeans")
    embOptions = EssCl.EmbedOptions(n_components=4)
    cons_cluster_2 = EssCl.ensemble_cspa(Prop_m, genes, cspa = CSPAoptions, embed= embOptions, save_html_to=directory + "/Results/File_1/Essem_CSPA.html")
    cons_cluster_3 = EssCl.ensemble_plurality_voting(Matrix, Prop_m, genes, embed=embOptions ,save_html_to=directory + "/Results/File_1/Essem_PV.html")
    end_time = time.time()
    print(f"Tiempo de ejecución (Agrupamiento otros consensos) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Comparación de composición de soluciones (JACCARD).
    start_time = time.time()
    Jaccard = JV.jaccard_index_solutions(Matrix)
    end_time = time.time()
    print(f"Tiempo de ejecución (Valores Jaccard) : {end_time - start_time:.6f} segundos")
    Heat.plot_html_heatmap(Jaccard, save_filepath= directory + "/Results/File_1/JaccardS.html",
                        x_label='Solution',
                        y_label='Solution',
                        title='Similitud de Jaccard entre soluciones',
                        z_label="Jaccard",
                        tooltip_format="Solution_ID_1: %{x}<br>Solution_ID_2: %{y}<br>Jaccard: %{z:.2f}")
    Ac.save_matrix(Jaccard,  directory + "/Results/File_1/jacca_matrix.csv",options)

    ###################################################################################################### Comparación de composición de clusters (RAND).
    start_time = time.time()
    Rand = RV.rand_index_solutions(Matrix)
    end_time = time.time()
    print(f"Tiempo de ejecución (Valores RI) : {end_time - start_time:.6f} segundos")
    Ac.save_matrix(Rand,  directory + "/Results/File_1/rand_matrix.csv",options)

    ###################################################################################################### Comparación de composición de clusters (ARI).
    start_time = time.time()
    Rand = RV.adjusted_rand_index_solutions(Matrix)
    end_time = time.time()
    print(f"Tiempo de ejecución (Valores ARI) : {end_time - start_time:.6f} segundos")
    Ac.save_matrix(Rand,  directory + "/Results/File_1/adj_rand_matrix.csv",options)

    ###################################################################################################### Reformateo de solución (Solucion-Cluster).  
    start_time = time.time()
    SC_matrix = SCM.solution_cluster_matrix(Matrix, genes, parallel=True, max_workers=8)
    end_time = time.time()
    print(f"Tiempo de ejecución (SCM) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Obtener clusters equivalentes (Jaccard).  
    start_time = time.time()
    df_equivalentes_1 = JV.find_equivalent_clusters_jaccard(SC_matrix)
    end_time = time.time()
    print(f"Tiempo de ejecución (grupos equivalentes J) : {end_time - start_time:.6f} segundos")
    Ac.save_dataframe(df_equivalentes_1, directory + "/Results/File_1/jacc_Equivalentes.csv")

    ###################################################################################################### Obtener clusters equivalentes (RI).  
    start_time = time.time()
    df_equivalentes = RV.find_equivalent_clusters_rand(SC_matrix)
    end_time = time.time()
    print(f"Tiempo de ejecución (grupos equivalentes RI) : {end_time - start_time:.6f} segundos")
    Ac.save_dataframe(df_equivalentes, directory + "/Results/File_1/Rand_Equivalentes.csv")

    ###################################################################################################### Obtener clusters equivalentes (ARI).  
    start_time = time.time()
    df_equivalentes = RV.find_equivalent_clusters_rand(SC_matrix, metric="adjusted_rand")
    end_time = time.time()
    print(f"Tiempo de ejecución (grupos equivalentes ARI) : {end_time - start_time:.6f} segundos")
    Ac.save_dataframe(df_equivalentes, directory + "/Results/File_1/A_Rand_Equivalentes.csv")

    ###################################################################################################### Obtener simbolos canonicos.
    start_time = time.time()
    EntrezID = ME.Convert_To_Entrez_ID(genes)
    end_time = time.time()
    print(f"Tiempo de ejecución (EntrezID Python) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Go Enrichment.
    start_time = time.time()
    GO_DF_P = GOeP.GoEnrichment(EntrezID,organism='hsapiens')
    Ac.save_dataframe(GO_DF_P,directory + "/Results/File_1/Enrichment_Example_Python.csv")
    end_time = time.time()
    print(f"Tiempo de ejecución (Enriquecimiento biologico con entrezID Python) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Enterz ID to Term.
    start_time = time.time()
    En_to_Ant = GOeP.AnnotationFromEntrezIDs(EntrezID,organism='hsapiens')
    end_time = time.time()
    print(f"Tiempo de ejecución (Entrez a Term) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Distancia de Wang entre los genes.
    symbols = WI.entrez_to_symbol_ncbi(entrez_ids=EntrezID, gene_info_path=r"C:\Users\benja\Desktop\workspace\ItalianEdge\Homo_sapiens.gene_info")
    symbols[symbols.index("MALAT1")] = "URS000001C914_9606"
    symbols[symbols.index("C1orf56")] = "MENT"

    start_time = time.time()
    order, Wang_Index = WI.compute_gene_similarity_matrix_go3(symbols, "goa_human")
    end_time = time.time()
    print(f"Tiempo de ejecución (Indice de Wang) : {end_time - start_time:.6f} segundos")
    
    ###################################################################################################### Dual Heatmap.
    start_time = time.time()
    wang_s, df_mod = WI.solution_wang_similarity_from_dataframe(genes, Wang_Index, df_equivalentes_1, SC_matrix)
    end_time = time.time()
    print(f"Tiempo de ejecución (Matriz dual) : {end_time - start_time:.6f} segundos")
    Ac.save_dataframe(df_mod, directory + "/Results/File_1/Equivalentes_con_wang.csv")
    Heat.plot_dual_heatmap_two_colors(Jaccard, wang_s, directory + "/Results/File_1/Dual.html")

    ###################################################################################################### Go Graph.
    EntrezID_P = ME.Convert_To_Entrez_ID(list(SC_matrix[0][1]),organism_gp='hsapiens', taxID=9606)
    GO_DF_P = GOeP.GoEnrichment(EntrezID_P)
    GtoT = GOeP.AnnotationFromEntrezIDs(EntrezID_P, Ontology=['GO:BP'], organism='hsapiens')
    term_pvalues = GO_DF_P.set_index("native")["p_value"].to_dict()
    Gplot.plot_gene_ratio(GO_DF_P, directory + "/Results/File_1/GR.html")
    Gplot.plot_qscore(GO_DF_P, directory + "/Results/File_1/QS.html")
    options_net = Gnet.GoNetworkOptions(min_genes_per_term=10)
    Gnet.plot_go_interaction_network_html(GtoT, term_pvalues, 
                                          gaf_path=r"C:\Users\benja\Desktop\workspace\ItalianEdge\goa_human",
                                          obo_path=r"C:\Users\benja\Desktop\workspace\ItalianEdge\go-basic.obo",
                                          options= options_net,
                                          save_html_to = directory + "/Results/File_1/Net.html")
    net_options = GHnet.GoHierarchyOptions(ontology="BP", min_genes_per_term=10, obo_path=r"go-basic.obo")
    GHnet.plot_go_hierarchy_html(GtoT, 
                                 term_pvalues,
                                 options=net_options,
                                 save_html_to = directory + "/Results/File_1/Tree.html")
    GCD.plot_go_chord_html(
    GtoT,
    save_html_to="go_chord.html"
    )


    ###################################################################################################### Threshold.
    threshold = ST.estimate_similarity_threshold(
        df_mod,
        column="Jaccard Similarity",
        plot=True,
        save_png_to="results/gmm_threshold.png"
    )
    print(threshold)

    RC.plot_similarity_raincloud_html(
        df_mod,
        column="Jaccard Similarity",
        save_html_to="similarity_raincloud.html"
    )  