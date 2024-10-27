######### Libraries #########
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
import pandas as pd

######### Functions #########

def He_clustering(ProportionMatrix, num_groups=4, dendrogram_file="dendrogram.png"):
    """
    He_clustering(function)
        Input:
            - ProportionMatrix: Result from 'ProportionMatrix_Similarity'
            function that creates a new solution.
            - num_groups: number of cluster for the solution.
        Output:
            - ConsensusSolution: Consensus cluster for comparitions.
        
        Description: Creation of the consensus cluster thar represents the
        performances of all solutions.
    """
    try:
        # Transformar la matriz en formato comprimido para el clustering jerárquico
        condensed_distance = squareform(ProportionMatrix)
        
        # Crear el dendrograma y guardarlo en un archivo
        linked = linkage(condensed_distance, method='average')
        plt.figure(figsize=(10, 7))
        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title("Dendrograma de Clustering Jerárquico")
        plt.xlabel("Muestras de datos")
        plt.ylabel("Distancia")
        plt.savefig(dendrogram_file)  # Guardar el dendrograma en un archivo
        plt.close()  # Cerrar la figura para liberar memoria
        print(f"Dendrograma guardado en '{dendrogram_file}'")

        # Aplicar el clustering jerárquico
        hc = AgglomerativeClustering(n_clusters=num_groups, metric='euclidean', linkage='average')
        ConsensusSolution = hc.fit_predict(ProportionMatrix)

        return ConsensusSolution

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None