######### Libraries #########
import unittest
import sys
import os
import numpy as np

######### Module Path #########
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from He_Clustering import (
    He_clustering
)

class TestHeClustering(unittest.TestCase):

    def test_clustering_output_length(self):
        # Matriz de distancia de ejemplo
        ProportionMatrix = np.array([
            [0.0, 1.0, 1.5, 0.5],
            [1.0, 0.0, 1.2, 1.8],
            [1.5, 1.2, 0.0, 2.5],
            [0.5, 1.8, 2.5, 0.0]
        ])
        num_groups = 2

        # Llamada a la función
        y_hc = He_clustering(ProportionMatrix, num_groups)

        # Verificar que la salida no sea None
        self.assertIsNotNone(y_hc, "La función devolvió None en lugar de una matriz de etiquetas")

        # Verificar que la longitud de la salida es correcta
        self.assertEqual(len(y_hc), ProportionMatrix.shape[0], "La longitud de las etiquetas no coincide con el número de muestras")

    def test_invalid_input(self):
        # ProportionMatrix no cuadrada para probar la excepción
        ProportionMatrix = np.array([
            [0.0, 1.0, 1.5],
            [1.0, 0.0, 1.2]
        ])

        # Llamada a la función con entrada inválida
        y_hc = He_clustering(ProportionMatrix, num_groups=2)

        # Verificar que la salida sea None cuando la entrada es incorrecta
        self.assertIsNone(y_hc, "La función no manejó correctamente la matriz no cuadrada")

if __name__ == "__main__":
    unittest.main()