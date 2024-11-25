######### Libraries #########
import unittest
import sys
import os
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import csr_matrix

######### Module Path #########
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from ProportionMatrix import (
    ProportionsMatrix
)

class TestProportionsMatrix(unittest.TestCase):

    def setUp(self):
        """
        Configuración de datos para las pruebas.
        """
        # Matriz sumada de ejemplo en formato CSR
        self.summed_matrix = csr_matrix([
            [10, 2, 4],
            [2, 8, 1],
            [4, 1, 6]
        ])
        self.total_solutions = 10
        
        # Matriz de presencia de genes (para count_presence = 1)
        self.presence_matrix = [
            [1, None, 1],
            [1, 1, None],
            [None, None, 1]
        ]
        self.presence_denominator = np.array([
            [2, 1, 1],
            [1, 1, 0],
            [1, 0, 2]
        ])  # Esto debe calcularse si tienes una función para ello.

    def test_valid_proportion_without_presence(self):
        """
        Prueba con parámetros válidos y count_presence = 0.
        """
        proportion_matrix, distance_matrix = ProportionsMatrix(self.summed_matrix, self.total_solutions)
        
        # Verificar los valores esperados
        expected_proportion = np.array([
            [1.0, 0.2, 0.4],
            [0.2, 0.8, 0.1],
            [0.4, 0.1, 0.6]
        ])
        expected_distance = 1 - expected_proportion

        np.testing.assert_array_almost_equal(proportion_matrix, expected_proportion)
        np.testing.assert_array_almost_equal(distance_matrix, expected_distance)

    def test_valid_proportion_with_presence(self):
        """
        Prueba con parámetros válidos y count_presence = 1.
        """
        proportion_matrix, distance_matrix = ProportionsMatrix(
            self.summed_matrix,
            self.total_solutions,
            Matrix=self.presence_matrix,
            count_prescence=1
        )
        
        # Compara las proporciones esperadas basadas en la matriz de presencia
        with np.errstate(divide='ignore', invalid='ignore'):
            expected_proportion = self.summed_matrix.toarray() / self.presence_denominator
            expected_proportion[np.isinf(expected_proportion)] = 0
        expected_distance = 1 - expected_proportion

        np.testing.assert_array_almost_equal(proportion_matrix, expected_proportion)
        np.testing.assert_array_almost_equal(distance_matrix, expected_distance)

    def test_invalid_summed_matrix_type(self):
        """
        Prueba con un tipo inválido para summed_matrix.
        """
        with self.assertRaises(TypeError):
            ProportionsMatrix(
                summed_matrix=[[10, 2, 4]],  # No es CSR
                total_solutions=self.total_solutions
            )

    def test_invalid_total_solutions(self):
        """
        Prueba con un valor inválido para total_solutions.
        """
        with self.assertRaises(ValueError):
            ProportionsMatrix(self.summed_matrix, total_solutions=0)

if __name__ == '__main__':
    unittest.main()