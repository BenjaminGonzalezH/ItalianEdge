import unittest
import numpy as np

from ParetoInsight_CPU.RandValues import (
    RandIndexSolutions
)

class TestRandIndexSolutions(unittest.TestCase):

    def setUp(self):
        # Tres soluciones distintas de agrupamiento de 4 genes
        self.matrix = np.array([
            [0, 0, 1, 1],  # Solución 1: grupos [0,1], [2,3]
            [1, 1, 0, 0],  # Solución 2: grupos invertidos respecto a la 1
            [0, 1, 0, 1]   # Solución 3: grupos alternados
        ])

    def test_regular_behavior(self):
        rand = RandIndexSolutions(self.matrix, n_threads=2)
        # La matriz debe ser simétrica
        self.assertTrue(np.allclose(rand, rand.T))
        # Autocomparación Rand siempre 1
        self.assertTrue(np.allclose(np.diag(rand), 1.0))
        # Debe tener mismo número de filas y columnas que soluciones
        self.assertEqual(rand.shape, (3, 3))
        # El índice Rand es un número entre 0 y 1 para cada caso
        self.assertTrue(np.all((rand >= 0) & (rand <= 1)))

    def test_empty_matrix(self):
        with self.assertRaises(RuntimeError):
            RandIndexSolutions(np.empty((0, 4)), n_threads=2)

    def test_insufficient_columns(self):
        with self.assertRaises(RuntimeError):
            RandIndexSolutions(np.zeros((2, 1)), n_threads=2)

    def test_identical_solutions(self):
        # Si todas las soluciones son iguales, Rand debe ser 1 fuera y dentro de la diagonal
        matrix = np.tile([2, 2, 1, 1], (2, 1))
        rand = RandIndexSolutions(matrix, n_threads=2)
        self.assertTrue(np.allclose(rand, 1.0))

    def test_completely_different_solutions(self):
        # Soluciones completamente separadas: cada gen en grupo diferente
        matrix = np.array([
            [0, 1, 2, 3],
            [3, 2, 1, 0]
        ])
        rand = RandIndexSolutions(matrix, n_threads=2)
        self.assertTrue(np.all(rand <= 1.0))
        self.assertTrue(np.all(rand >= 0.0))
        self.assertTrue(np.allclose(np.diag(rand), 1.0))

if __name__ == "__main__":
    unittest.main()
