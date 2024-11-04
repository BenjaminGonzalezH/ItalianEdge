# Importaciones.
import sys
import os
import numpy as np
from scipy.sparse import csr_matrix

# Librerias propias.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'App\CoMOcG', 'src')))
from ReadSolution import ReadInputCSV
from ProportionMatrix import (
    connectivityMatrix,
    sum_connectivity_matrices,
    ProportionMatrix_Similarity,
    ProportionMatrix_Disimilarity
)
from He_Clustering import He_clustering
from SolutionClusterMatrix import (
    SolutionClusterMatrix_GeneID
)



# Data.
csv_path_1 = r"C:\Users\benja\OneDrive\Escritorio\WorkSpace\ItalianEdge\id_y_cluster_GSE10797_10600genes_exp03.csv"
csv_path_2 = r"C:\Users\benja\OneDrive\Escritorio\WorkSpace\ItalianEdge\id_y_cluster_GSE6919_U95C_3444genes_exp01.csv"

# Procesamiento.
genes, num_genes, Matrix = ReadInputCSV(csv_path_1, n_threads=8)
genes.pop(0)

con_m = connectivityMatrix(Matrix.tolist(), max_workers=8)
sum_m = sum_connectivity_matrices(con_m)
pro_m = ProportionMatrix_Similarity(sum_m, total_solutions=25)
pro_m_d = ProportionMatrix_Disimilarity(pro_m)

cluster = He_clustering(pro_m_d.toarray().tolist(), genes, 4, "dendograma.png", 0)
print(list(map(int, cluster)))
