# Importaciones.
import sys
import os
import numpy as np
from scipy.sparse import csr_matrix
import csv
import pandas as pd
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

# Librerias propias.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'App\CoMOcG', 'src')))
import ReadSolution as RD
import ConnectivityMatrix as CM
import ProportionMatrix as PM
import He_Clustering as He
import SolutionClusterMatrix as SCM
import JaccardValues as JV
import SolutionComposition as SC
import GoEnrischment as GOe
import Go_Plots as plots

# Obtain test files.
file_1 = r"C:\Users\benja\Desktop\workspace\ItalianEdge\test_files_reals\archivo_prueba_1_116_500.csv"
file_2 = r"C:\Users\benja\Desktop\workspace\ItalianEdge\test_files_reals\archivo_prueba_2_116_3444.csv"
file_3 = r"C:\Users\benja\Desktop\workspace\ItalianEdge\test_files_reals\archivo_prueba_3_25_133.csv"

start_time = time.time()
genes, num_genes, Matrix  = RD.ReadInputCSV_threads(file_1, n_workers = 8, solutions_id_colum=1)
end_time = time.time()
print(f"Tiempo de ejecución (lectura) : {end_time - start_time:.6f} segundos")

start_time = time.time()
connec_sum = CM.connectivityMatrix_threads(Matrix,8)
connec_sum = CM.sum_connectivity_matrices(connec_sum)
#CM.save_connectivity_matrix_as_csv(connec_sum,"C:/Users/benja/Desktop/workspace/ItalianEdge/Results/Connectivity_Matrix_file_1.csv")
end_time = time.time()
print(f"Tiempo de ejecución (conectividad) : {end_time - start_time:.6f} segundos")

start_time = time.time()
Prop_m, Dist_m = PM.ProportionsMatrix(connec_sum)
#PM.save_matrices(Prop_m, Dist_m,
#                 "C:/Users/benja/Desktop/workspace/ItalianEdge/Results/Prop_matrix.csv",
#                 "C:/Users/benja/Desktop/workspace/ItalianEdge/Results/Dist_matrix.csv")
end_time = time.time()
print(f"Tiempo de ejecución (Proporcion y distancia) : {end_time - start_time:.6f} segundos")
#PM.plot_and_save_heatmaps(Prop_m, Dist_m, 
#                          save_path="C:/Users/benja/Desktop/workspace/ItalianEdge/Results")

"""start_time = time.time()
cons_cluster = He.He_clustering(Dist_m, genes, 4, 
                                save_path="C:/Users/benja/Desktop/workspace/ItalianEdge/Results",
                                dendrogram_file="Dendogram_file_1.png")
end_time = time.time()
print(f"Tiempo de ejecución (Agrupamiento Jerarquico) : {end_time - start_time:.6f} segundos")"""

start_time = time.time()
SC_matrix = SCM.SolutionClusterMatrix_GeneID(Matrix, genes, 8)
end_time = time.time()
print(f"Tiempo de ejecución (Cambio de estructura) : {end_time - start_time:.6f} segundos")

start_time = time.time()
Jaccard = JV.process_JaccardValues(Matrix, 8)
end_time = time.time()
print(f"Tiempo de ejecución (Valores Jaccard) : {end_time - start_time:.6f} segundos")
#JV.plot_jaccard_heatmap(Jaccard, save_path="C:/Users/benja/Desktop/workspace/ItalianEdge/Results",resolution=600,figsize=(20,16))
#JV.save_jaccard_matrix(Jaccard, "C:/Users/benja/Desktop/workspace/ItalianEdge/Results/JaccadValues")

start_time = time.time()
Jaccard_E = JV.Jaccar_similarityClusters(SC_matrix[0], SC_matrix[1])
end_time = time.time()
print(f"Tiempo de ejecución (Comparar dos soluciones Jaccard) : {end_time - start_time:.6f} segundos")
#print(Jaccard_E)
#JV.plot_jaccard_heatmap(Jaccard_E, save_path="C:/Users/benja/Desktop/workspace/ItalianEdge/Results")

start_time = time.time()
Coposition = SC.process_proportion_genessolution(Matrix, 8)
print(f"Tiempo de ejecución (Composición de composición) : {end_time - start_time:.6f} segundos")
#print(Coposition)
#JV.save_jaccard_matrix(Jaccard, "C:/Users/benja/Desktop/workspace/ItalianEdge/Results/composition.csv")
#JV.plot_jaccard_heatmap(Jaccard, title="Composition of functions" ,save_path="C:/Users/benja/Desktop/workspace/ItalianEdge/Results",
#                        resolution=600,figsize=(20,16))

start_time = time.time()
Coposition_E = SC.AmountGenes_Equals(SC_matrix[0], SC_matrix[1])
end_time = time.time()
print(f"Tiempo de ejecución (Comparar dos soluciones cant_genes) : {end_time - start_time:.6f} segundos")
#print(Coposition_E)
#JV.plot_jaccard_heatmap(Coposition_E, title="Composition of functions" ,save_path="C:/Users/benja/Desktop/workspace/ItalianEdge/Results",
#                        resolution=600,figsize=(20,16))

#start_time = time.time()
#GOe.setup_r_environment()
#end_time = time.time()
#print(f"Tiempo de ejecución (Cargar entorno de R) : {end_time - start_time:.6f} segundos")

start_time = time.time()
df_enrichment = GOe.perform_go_enrichment(list(SC_matrix[0][3]))
end_time = time.time()
print(f"Tiempo de ejecución (Enriquecimiento con términos) : {end_time - start_time:.6f} segundos")

start_time = time.time()
df_enrichment_wang = GOe.perform_wang_similarity(df_enrichment)
end_time = time.time()
print(f"Tiempo de ejecución (Calculo de distancia de Wang) : {end_time - start_time:.6f} segundos")
GOe.save_results(df_enrichment_wang,"C:/Users/benja/Desktop/workspace/ItalianEdge/Results/wang.csv" )

start_time = time.time()
genes_to_go = plots.map_genes_to_go_terms(df_enrichment_wang)
plots.generate_heatmap_from_genes(df_enrichment_wang, genes_to_go)
plots.plot_gene_ratio(df_enrichment_wang)
plots.plot_qscore(df_enrichment_wang)
#plots.plot_go_interaction_network(df_enrichment_wang, threshold=0.2)
plots.plot_go_tree_rpy2(df_enrichment_wang, save_path="C:/Users/benja/Desktop/workspace/ItalianEdge/Results/tree.pdf")
end_time = time.time()
print(f"Tiempo de ejecución (graficos) : {end_time - start_time:.6f} segundos")