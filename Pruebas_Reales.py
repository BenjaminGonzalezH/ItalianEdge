# Importaciones.
import sys
import os
import numpy as np
from scipy.sparse import csr_matrix
import csv
import pandas as pd
import numpy as np
import time
from goatools.obo_parser import GODag
from goatools.associations import read_ncbi_gene2go
import pandas as pd

# Librerias propias.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'App\CoMOcG', 'src')))
import ReadSolution as RD
import ConnectivityMatrix as CM
import ProportionMatrix as PM
import He_Clustering as He
import SolutionClusterMatrix as SCM
import ProportionMatrixSC as PMSC
import JaccardValues as JV
import GoEnrischment as GOe
import Go_Plots as plots

# Obtain test files.
file_1 = r"C:\Users\benja\OneDrive\Escritorio\Workspace\ItalianEdge\test_files_reals\archivo_prueba_1_116_500.csv"
file_2 = r"C:\Users\benja\OneDrive\Escritorio\Workspace\ItalianEdge\test_files_reals\archivo_prueba_2_116_3444.csv"
file_3 = r"C:\Users\benja\OneDrive\Escritorio\Workspace\ItalianEdge\test_files_reals\archivo_prueba_3_25_133.csv"

start_time = time.time()
genes, num_genes, Matrix  = RD.ReadInputCSV_threads(file_1, n_workers = 8, solutions_id_colum=1)
#genes.pop(0)
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
Cluster = He.He_clustering(Disimilar, genes, num_groups=4, 
                           dendrogram_file= "dendograma.png", show_flag=0)
end_time = time.time()
print(f"Tiempo de ejecución (Cluster) : {end_time - start_time:.6f} segundos")

np.append(Matrix, np.array(Cluster))
start_time = time.time()
SolutionClusterMatrix = SCM.SolutionClusterMatrix_GeneID(Matrix, genes, max_workers=8)
end_time = time.time()
print(f"Tiempo de ejecución (SolutionClusterMatrix) : {end_time - start_time:.6f} segundos")

"""start_time = time.time()
JVMatrix = JV.process_JaccardValues(Matrix, n_threads=8)
Distance = 1-JVMatrix
end_time = time.time()
print(f"Tiempo de ejecución (Jaccard) : {end_time - start_time:.6f} segundos")
print(Distance)

start_time = time.time()
JMatrix = JV.Jaccar_AmountGenes(SolutionClusterMatrix[0], SolutionClusterMatrix[1])
end_time = time.time()
print(f"Tiempo de ejecución (AmountGenes) : {end_time - start_time:.6f} segundos")
print(JMatrix)"""

start_time = time.time()
GOe.setup_r_environment()
end_time = time.time()
print(f"Tiempo de ejecución (Set R) : {end_time - start_time:.6f} segundos")

#start_time = time.time()
#entrezID = GOe.convert_symbols_to_entrez(genes)
#end_time = time.time()
#print(f"Tiempo de ejecución (ConverID) : {end_time - start_time:.6f} segundos")
#print(entrezID)

start_time = time.time()
df = GOe.perform_go_enrichment(list(SolutionClusterMatrix[0][0]), convert_ids=True)
end_time = time.time()
print(f"Tiempo de ejecución (Enrichment) : {end_time - start_time:.6f} segundos")
#print(df)
#print(df.columns)
#print(df['geneID'])

start_time = time.time()
wang = GOe.perform_wang_similarity(df)
end_time = time.time()
print(f"Tiempo de ejecución (Wang) : {end_time - start_time:.6f} segundos")
#print(wang)

start_time = time.time()
genes_terms = plots.map_genes_to_go_terms(df)
end_time = time.time()
print(f"Tiempo de ejecución (Dict_Genes) : {end_time - start_time:.6f} segundos")
#print(genes_terms[161742])

start_time = time.time()
plots.generate_heatmap_from_genes(wang,genes_terms)
end_time = time.time()
print(f"Tiempo de ejecución (Plot) : {end_time - start_time:.6f} segundos")