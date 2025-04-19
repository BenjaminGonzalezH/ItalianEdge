######### Libraries #########
import unittest                         # Test interface.
import numpy as np                      # Numbers ADT managment.
from scipy.sparse import csr_matrix     # Compresed matrix version.
from CoMOcG.ConnectivityMatrix import (
    compute_connectivity,
    connectivityMatrix,
    sum_connectivity_matrices
)

class TestConnectivityMatrix(unittest.TestCase):

    def test_compute_connectivity_basic(self):
        solution = np.array([1, 2, 1, 2])
        expected = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ])
        result = compute_connectivity(solution).toarray()
        np.testing.assert_array_equal(result, expected)

    def test_compute_connectivity_empty(self):
        with self.assertRaises(ValueError):
            compute_connectivity(np.array([]))

    def test_connectivityMatrix_threads(self):
        solutions = [np.array([1, 2, 1, 2]), np.array([1, 1, 2, 2])]
        matrices = connectivityMatrix(solutions, n_threads=2)
        self.assertEqual(len(matrices), 2)
        self.assertTrue(all(isinstance(m, csr_matrix) for m in matrices))

    def test_connectivityMatrix_invalid_threads(self):
        solutions = [np.array([1, 2]), np.array([1, 1])]
        with self.assertRaises(ValueError):
            connectivityMatrix(solutions, n_threads=0)

    def test_sum_connectivity_matrices(self):
        m1 = csr_matrix(np.array([
            [1, 0],
            [0, 1]
        ]))
        m2 = csr_matrix(np.array([
            [0, 1],
            [1, 0]
        ]))
        result = sum_connectivity_matrices([m1, m2]).toarray()
        expected = np.array([
            [1, 1],
            [1, 1]
        ])
        np.testing.assert_array_equal(result, expected)

    def test_sum_connectivity_matrices_empty(self):
        with self.assertRaises(ValueError):
            sum_connectivity_matrices([])

    def test_sum_connectivity_matrices_shape_mismatch(self):
        m1 = csr_matrix(np.ones((2, 2)))
        m2 = csr_matrix(np.ones((3, 3)))
        with self.assertRaises(ValueError):
            sum_connectivity_matrices([m1, m2])

if __name__ == '__main__':
    unittest.main()