######### Libraries #########
import unittest
import sys
import os
import numpy as np
import io
from contextlib import redirect_stdout

######### Module Path #########
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from He_Clustering import (
    He_clustering
)

class TestHeClustering(unittest.TestCase):

    def test_clustering_output_length(self):
        # Matriz de distancia de ejemplo
        ProportionMatrix = np.array([
            [0.0, 1.0, 0.5, 0.5],
            [1.0, 0.0, 0.2, 0.8],
            [0.5, 0.2, 0.0, 0.5],
            [0.5, 0.8, 0.5, 0.0]
        ])
        num_groups = 4
        genes = ['1', '2', '3', '4']
        expected_result = [1, 1, 1, 1]

        # Llamada a la función
        cluster = He_clustering(ProportionMatrix, genes, num_groups, "dendograma.png", 0)
        self.assertEqual(cluster,expected_result)


    def test_invalid_input(self):
        # ProportionMatrix no cuadrada para probar la excepción
        ProportionMatrix = np.array([
            [0.0, 1.0, 1.5],
            [1.0, 0.0, 1.2]
        ])
        genes = ['1', '2', '3', '4']
        num_groups = 4

        # Not using the print message.
        f = io.StringIO()
        with redirect_stdout(f):
            #RUN.
            cluster = He_clustering(ProportionMatrix, genes, num_groups, "dendograma.png", 0)
        # Take message.
        mensaje_impreso = f.getvalue()

        # Verificar que la salida sea None cuando la entrada es incorrecta
        self.assertIsNone(cluster, "La función no manejó correctamente la matriz no cuadrada")

if __name__ == "__main__":
    unittest.main()