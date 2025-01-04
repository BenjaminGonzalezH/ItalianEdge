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
CM.save_connectivity_matrix_as_csv(connec_sum,"C:/Users/benja/Desktop/workspace/ItalianEdge/Results/Connectivity_Matrix_file_1.csv")
end_time = time.time()
print(f"Tiempo de ejecución (conectividad) : {end_time - start_time:.6f} segundos")

start_time = time.time()
Prop_m, Dist_m = PM.ProportionsMatrix(connec_sum)
PM.save_matrices(Prop_m, Dist_m,
                 "C:/Users/benja/Desktop/workspace/ItalianEdge/Results/Prop_matrix.csv",
                 "C:/Users/benja/Desktop/workspace/ItalianEdge/Results/Dist_matrix.csv")
end_time = time.time()
print(f"Tiempo de ejecución (conectividad) : {end_time - start_time:.6f} segundos")
PM.plot_and_save_heatmaps(Prop_m, Dist_m, 
                          save_path="C:/Users/benja/Desktop/workspace/ItalianEdge/Results")

start_time = time.time()
cons_cluster = He.He_clustering(Dist_m, genes, 4, 
                                save_path="C:/Users/benja/Desktop/workspace/ItalianEdge/Results",
                                dendrogram_file="Dendogram_file_1.png")
end_time = time.time()
print(f"Tiempo de ejecución (conectividad) : {end_time - start_time:.6f} segundos")

start_time = time.time()

end_time = time.time()
print(f"Tiempo de ejecución (conectividad) : {end_time - start_time:.6f} segundos")