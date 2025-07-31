if __name__ == "__main__":
    # Importaciones.
    import os
    import time

    # Librerias propias CPU.
    import App.ParetoInsight_CPU.ReadSolution as RD
    import App.ParetoInsight_CPU.Actions as Ac_CPU
    import App.ParetoInsight_CPU.Heatmaps as Heat
    import App.ParetoInsight_CPU.He_Clustering as He

    # Librerias propias GPU.
    import App.ParetoInsight_GPU.Actions as AC
    import App.ParetoInsight_GPU.ConsensusMatrix as CM
    import App.ParetoInsight_GPU.JaccardValues as JV
    import App.ParetoInsight_GPU.RandValues as RV
    

    # Obtain actual directory.
    directory = os.path.dirname(__file__)

    # Obtain test files.
    file_1 = directory + r"\test_files_reals\archivo_prueba_1_116_500.csv"
    file_2 = directory + r"\test_files_reals\archivo_prueba_2_116_3444.csv"
    file_3 = directory + r"\test_files_reals\archivo_prueba_3_25_133.csv"

    # GPU pool initialization.
    MemPool = AC.GPU_MemoryPool(limit_bytes=0)

    ###################################################################################################### Lectura de archivo.
    start_time = time.time()
    Matrix, genes  = RD.ReadSolutionsFile(file_2,"csv")
    Matrix = AC.TransformMathStructure(Matrix)
    end_time = time.time()
    print(f"Tiempo de ejecución (lectura) : {end_time - start_time:.6f} segundos")

    ###################################################################################################### Matriz de consenso.
    start_time = time.time()
    Prop_m, Dist_m = CM.ConsensusMatrix(Matrix)
    Prop_m = AC.TransformMathStructure(Prop_m,"GPU")
    Dist_m = AC.TransformMathStructure(Dist_m,"GPU")
    end_time = time.time()
    print(f"Tiempo de ejecución (consenso) : {end_time - start_time:.6f} segundos")
    Ac_CPU.save_matrix(Prop_m,  directory + "/Results/File_2_GPU/Prop_matrix.csv")
    Ac_CPU.save_matrix(Dist_m,  directory + "/Results/File_2_GPU/Dist_matrix.csv")
    Heat.plot_html_heatmap(Prop_m,  directory + "/Results/File_2_GPU/Prop_matrix.html",
                        x_label="Gen",
                        y_label="Gen",
                        title="Similitud entre genes basada en asignaciones de grupos",
                        z_label="Proporción de coincidencia",
                        tooltip_format="Gen_ID_1: %{x}<br>Gen_ID_2: %{y}<br>Proporción: %{z:.2f}")
    
    ###################################################################################################### Comparación de composición de soluciones (JACCARD).
    start_time = time.time()
    Jaccard = JV.JaccardIndexSolutions(Matrix)
    Jaccard = AC.TransformMathStructure(Jaccard,"GPU")
    end_time = time.time()
    print(f"Tiempo de ejecución (Valores Jaccard) : {end_time - start_time:.6f} segundos")
    Heat.plot_html_heatmap(Jaccard, save_filepath= directory + "/Results/File_2_GPU/JaccardS.html",
                        x_label='Solution',
                        y_label='Solution',
                        title='Similitud de Jaccard entre soluciones',
                        z_label="Jaccard",
                        tooltip_format="Solution_ID_1: %{x}<br>Solution_ID_2: %{y}<br>Jaccard: %{z:.2f}")
    Ac_CPU.save_matrix(Jaccard,  directory + "/Results/File_2_GPU/Jaccard_Matrix.csv")
    
    ###################################################################################################### Comparación de composición de clusters (RAND).
    start_time = time.time()
    Rand = RV.RandIndexSolutions(Matrix)
    end_time = time.time()
    print(f"Tiempo de ejecución (Valores RI) : {end_time - start_time:.6f} segundos")


    # GPU free memory.
    AC.GPU_freeMemoryPool(MemPool)