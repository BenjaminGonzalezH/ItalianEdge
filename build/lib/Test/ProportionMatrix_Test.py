######### Libraries #########
import unittest
import sys
import os
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import csr_matrix

######### Module Path #########
from CoMOcG.ProportionMatrix import (
    ProportionsMatrix
)

class TestProportionsMatrix(unittest.TestCase):

    def test_valid_input(self):
        matrix = np.array([
            [3, 2, 0],
            [2, 3, 1],
            [0, 1, 3]
        ])
        csr = csr_matrix(matrix)
        proportion, distance = ProportionsMatrix(csr)

        expected_proportion = matrix / 3  # 3 es el valor en [0,0]
        expected_distance = 1 - expected_proportion

        np.testing.assert_array_almost_equal(proportion, expected_proportion, decimal=5)
        np.testing.assert_array_almost_equal(distance, expected_distance, decimal=5)

    def test_not_csr_matrix(self):
        matrix = np.array([
            [1, 0],
            [0, 1]
        ])
        with self.assertRaises(TypeError):
            ProportionsMatrix(matrix)

    def test_zero_division_handling(self):
        matrix = np.array([
            [0, 2],
            [2, 0]
        ])
        csr = csr_matrix(matrix)
        proportion, distance = ProportionsMatrix(csr)

        expected_proportion = np.zeros_like(matrix, dtype=float)
        expected_distance = np.ones_like(matrix, dtype=float)

        np.testing.assert_array_equal(proportion, expected_proportion)
        np.testing.assert_array_equal(distance, expected_distance)

    def test_output_shapes(self):
        matrix = np.array([
            [5, 2, 1],
            [2, 5, 3],
            [1, 3, 5]
        ])
        csr = csr_matrix(matrix)
        proportion, distance = ProportionsMatrix(csr)

        self.assertEqual(proportion.shape, matrix.shape)
        self.assertEqual(distance.shape, matrix.shape)

if __name__ == '__main__':
    unittest.main()