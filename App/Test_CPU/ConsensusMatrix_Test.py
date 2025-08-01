import unittest
import numpy as np

from ParetoInsight_CPU.ConsensusMatrix import (
    ConsensusMatrix
)

class TestConsensusMatrix(unittest.TestCase):

    def setUp(self):
        # Matriz ejemplo: 3 soluciones para 4 genes
        self.matrix = np.array([
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1]
        ])

    def test_regular_behavior(self):
        coincidence, consensus = ConsensusMatrix(self.matrix)
        self.assertEqual(coincidence.shape, (4, 4))
        self.assertEqual(consensus.shape, (4, 4))
        self.assertTrue(np.allclose(np.diag(coincidence), 1.0))
        self.assertTrue((coincidence >= 0).all() and (coincidence <= 1).all())
        self.assertTrue((consensus >= 0).all() and (consensus <= 1).all())

    def test_empty_matrix(self):
        with self.assertRaises(RuntimeError):
            ConsensusMatrix(np.empty((0, 4)))

    def test_insufficient_columns(self):
        with self.assertRaises(RuntimeError):
            ConsensusMatrix(np.zeros((3, 1)))

    def test_perfect_consensus(self):
        # Todas las soluciones iguales: coincidencia completa
        mat = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0]
        ])
        coincidence, consensus = ConsensusMatrix(mat)
        self.assertTrue(np.all(coincidence == np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1]
        ])))

    def test_no_shared_clusters(self):
        # Todas las soluciones diferentes (ningún par está en el mismo cluster)
        mat = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        coincidence, consensus = ConsensusMatrix(mat)
        self.assertTrue(np.allclose(coincidence, np.eye(3)))

if __name__ == "__main__":
    unittest.main()
