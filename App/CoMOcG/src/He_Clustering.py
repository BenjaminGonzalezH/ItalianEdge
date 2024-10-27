######### Libraries #########
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform

######### Functions #########

def He_clustering(ProportionMatrix, num_groups=4):
    """
    Realiza el clustering jerárquico y dibuja un dendrograma para una matriz de proporciones.
    
    Parámetros:
    - ProportionMatrix (ndarray): Matriz de distancia o proporciones entre muestras.
    - num_groups (int): Número de grupos para el clustering.
    
    Retorna:
    - y_hc (ndarray): Etiquetas de los clusters asignadas a cada muestra.
    """
    try:
        # Convertir la matriz cuadrada en formato comprimido
        condensed_distance = squareform(ProportionMatrix)
        
        # Crear el dendrograma
        linked = linkage(condensed_distance, method='average')
        plt.figure(figsize=(10, 7))
        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title("Dendrograma de Clustering Jerárquico")
        plt.xlabel("Muestras de datos")
        plt.ylabel("Distancia")
        plt.show()

        # Aplicar el clustering jerárquico
        hc = AgglomerativeClustering(n_clusters=num_groups, affinity='euclidean', linkage='average')
        y_hc = hc.fit_predict(ProportionMatrix)
        
        return y_hc

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None