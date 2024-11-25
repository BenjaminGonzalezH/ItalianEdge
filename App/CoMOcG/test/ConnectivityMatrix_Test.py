######### Libraries #########
import unittest
import sys
import os
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import csr_matrix
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from ConnectivityMatrix import (
    compute_connectivity,
    connectivityMatrix_threads,
    connectivityMatrix_processes,
    sum_connectivity_matrices
)

class TestConnectivityMatrix(unittest.TestCase):
    def setUp(self):
        self.solutions = [
            [1, 2, 1, 2],
            [3, 3, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 3, 3],
        ]
        self.expected_matrices = [
            csr_matrix(np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])),
            csr_matrix(np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])),
            csr_matrix(np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])),
            csr_matrix(np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])),
        ]

    def test_compute_connectivity(self):
        for solution, expected in zip(self.solutions, self.expected_matrices):
            result = compute_connectivity(solution)
            np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def test_connectivityMatrix_threads(self):
        result_matrices = connectivityMatrix_threads(self.solutions, n_threads=2)
        for result, expected in zip(result_matrices, self.expected_matrices):
            np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def test_connectivityMatrix_processes(self):
        result_matrices = connectivityMatrix_processes(self.solutions, n_jobs=2)
        for result, expected in zip(result_matrices, self.expected_matrices):
            np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            compute_connectivity([])
        with self.assertRaises(ValueError):
            connectivityMatrix_threads([], n_threads=0)
        with self.assertRaises(ValueError):
            connectivityMatrix_processes([], n_jobs=-1)

    def test_large_input(self):
        large_solutions = [[1, 2, 3, 4] * 100] * 10
        result_matrices = connectivityMatrix_threads(large_solutions, n_threads=4)
        self.assertEqual(len(result_matrices), len(large_solutions))
        result_matrices = connectivityMatrix_processes(large_solutions, n_jobs=4)
        self.assertEqual(len(result_matrices), len(large_solutions))

class TestSumConnectivityMatrices(unittest.TestCase):

    def setUp(self):
        """Configura las matrices de prueba."""
        # Matrices de conectividad de prueba
        self.matrices = [
            csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
            csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            csr_matrix([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
        ]
        self.expected_sum = csr_matrix([
            [2, 2, 1],
            [2, 2, 1],
            [1, 1, 2]
        ])

    def test_sum_correctly(self):
        """Prueba si la función suma correctamente las matrices."""
        result = sum_connectivity_matrices(self.matrices)
        np.testing.assert_array_equal(result.toarray(), self.expected_sum.toarray())

    def test_empty_list(self):
        """Prueba si la función maneja una lista vacía correctamente."""
        with self.assertRaises(ValueError):
            sum_connectivity_matrices([])

    def test_invalid_elements(self):
        """Prueba si la función detecta elementos no válidos en la lista."""
        invalid_matrices = self.matrices + [np.array([[1, 0], [0, 1]])]  # Agregar una matriz densa no válida.
        with self.assertRaises(TypeError):
            sum_connectivity_matrices(invalid_matrices)

    def test_single_matrix(self):
        """Prueba si la función retorna la misma matriz si hay una sola."""
        single_matrix = [csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]])]
        expected_matrix = csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        result = sum_connectivity_matrices(single_matrix)
        np.testing.assert_array_equal(result.toarray(), expected_matrix.toarray())

    def test_different_shapes(self):
        """Prueba si la función detecta matrices de diferentes tamaños."""
        invalid_matrices = [
            csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
            csr_matrix([[1, 1], [1, 1]])  # Diferente tamaño.
        ]
        with self.assertRaises(ValueError):
            sum_connectivity_matrices(invalid_matrices)

if __name__ == "__main__":
    unittest.main()