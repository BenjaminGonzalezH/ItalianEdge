# Importaciones.
import sys
import os
import numpy as np
from scipy.sparse import csr_matrix
import csv
import pandas as pd
import numpy as np
import time

# Librerias propias.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'App\CoMOcG', 'src')))
import ReadSolution as RD
import ConnectivityMatrix as CM
import ProportionMatrix as PM
import He_Clustering as He

# Obtain test files.
file_1 = r"C:\Users\Benjamin Gonzalez\Desktop\Workspace\ItalianEdge\test_files_reals\archivo_prueba_1_116_500.csv"
file_2 = r"C:\Users\Benjamin Gonzalez\Desktop\Workspace\ItalianEdge\test_files_reals\archivo_prueba_2_116_3444.csv"
file_3 = r"C:\Users\Benjamin Gonzalez\Desktop\Workspace\ItalianEdge\test_files_reals\archivo_prueba_3_25_133.csv"

start_time = time.time()
genes, num_genes, Matrix  = RD.ReadInputCSV_threads(file_2, n_workers = 8, solutions_id_colum=0)
genes.pop(0)
end_time = time.time()
print(f"Tiempo de ejecución (lectura) : {end_time - start_time:.6f} segundos")

start_time = time.time()
connectivity = CM.connectivityMatrix_threads(Matrix, n_threads = 8)
end_time = time.time()
print(f"Tiempo de ejecución (conectividad) : {end_time - start_time:.6f} segundos")

start_time = time.time()
sum_connectivity = CM.sum_connectivity_matrices(connectivity)
end_time = time.time()
print(f"Tiempo de ejecución (suma conectividad) : {end_time - start_time:.6f} segundos")

start_time = time.time()
Similarity, Disimilar = PM.ProportionsMatrix(sum_connectivity, 116)
end_time = time.time()
print(f"Tiempo de ejecución (Proporción) : {end_time - start_time:.6f} segundos")

start_time = time.time()
Cluster = He.He_clustering(Disimilar, genes, num_groups=4, show_flag=0)
end_time = time.time()
print(f"Tiempo de ejecución (Cluster) : {end_time - start_time:.6f} segundos")

#print(Cluster)
#print([i for i, value in enumerate(Cluster) if value != 1])