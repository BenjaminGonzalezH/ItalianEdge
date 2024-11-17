# Importaciones.
import sys
import os
import numpy as np
from scipy.sparse import csr_matrix
import csv
import pandas as pd
import numpy as np

# Librerias propias.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'App\CoMOcG', 'src')))
from ReadSolution import ReadInputCSV_NoID
from ReadSolution import ReadInputCSV
from ProportionMatrix import (
    connectivityMatrix,
    sum_connectivity_matrices,
    ProportionMatrix_Similarity,
    ProportionMatrix_Disimilarity,
    connectivity_matrices_function,
    consensus_matrix_function
)
from He_Clustering import (
    He_clustering,
    consensus_cluster_function
)
from SolutionClusterMatrix import (
    SolutionClusterMatrix_GeneID
)
from GoEnrischment import (
    enrich_go
)


# Data.
csv_path_1 = r"C:\Users\benja\OneDrive\Escritorio\WorkSpace\ItalianEdge\id_y_cluster_GSE10797_10600genes_exp03.csv"
csv_path_2 = r"C:\Users\benja\OneDrive\Escritorio\WorkSpace\ItalianEdge\archivo_prueba_1.csv"

# Procesamiento.
genes, num_genes, Matrix = ReadInputCSV_NoID(csv_path_2, n_threads=8)

"""for solution in Matrix:
    if(solution[88] == solution[117]):
        print(f"Sí, coinciden: {solution[88]} - {solution[117]}")"""
        

"""df = pd.read_csv("matriz_numerada.csv")
df_sin_primera_columna = df.iloc[:, 1:]
matriz = df_sin_primera_columna.values
#matriz_redondeada = np.round(matriz, 2)

for i in range(matriz.shape[0]):
    for j in range(i, matriz.shape[1]):
        if matriz[i, j] != matriz[j, i]:
            matriz[j, i] -= 1

matriz = matriz/25
matriz = 1-matriz
matriz[np.diag_indices(matriz.shape[0])] -= 1
print(matriz)
matriz[15,26] = matriz[15,26]*-1
matriz[26,15] = matriz[26,15]*-1
matriz[20,23] = matriz[20,23]*-1
matriz[23,20] = matriz[23,20]*-1"""

#SolutionClusterMatrix = SolutionClusterMatrix_GeneID(Matrix, genes)

con_m = connectivityMatrix(Matrix.tolist(), max_workers=8)
#con_m_1 = connectivity_matrices_function(Matrix, 116, Matrix.shape[1])
#sum_m_1 = consensus_matrix_function(con_m_1,Matrix.shape[1],116)
#cluster = consensus_cluster_function(sum_m_1,4)

sum_m = sum_connectivity_matrices(con_m)
pro_m = ProportionMatrix_Similarity(sum_m, total_solutions=117)
pro_m_d = ProportionMatrix_Disimilarity(pro_m)
np.fill_diagonal(pro_m_d, 0)

"""diferencias = np.where(sum_m.toarray().astype(int) != matriz)

# Convertir los índices a una lista de coordenadas
coordenadas_diferencias = list(zip(diferencias[0], diferencias[1]))

print("Índices donde los valores son diferentes:", coordenadas_diferencias)
print(sum_m.toarray()[0,1])
print(matriz[0,1])"""

cluster = He_clustering(pro_m_d.tolist(), genes, 4, "dendograma.png", 0)
print(list(map(int, cluster)))
#path_obo = r"C:\Users\benja\OneDrive\Escritorio\go-basic.obo"
#path_on = r"C:\Users\benja\OneDrive\Escritorio\gene2go"
#result = enrich_go(SolutionClusterMatrix[0][0],path_obo, path_on)
#print(result)