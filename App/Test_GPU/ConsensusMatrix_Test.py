import unittest
import cupy as cp
import numpy as np

from ParetoInsight_GPU.ConsensusMatrix import (
    ConsensusMatrixCupy
)

class TestConsensusMatrixCupy(unittest.TestCase):
    def setUp(self):
        # Tres soluciones de clustering para 4 genes
        # En CPU para fácil visualización
        self.cpu_mat = np.array([
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1]
        ])
        self.gpu_mat = cp.array(self.cpu_mat)

    def test_regular_behavior(self):
        coincidence, consensus = ConsensusMatrixCupy(self.gpu_mat)
        # Son matrices cuadradas y en GPU
        self.assertEqual(coincidence.shape, (4,4))
        self.assertIsInstance(coincidence, cp.ndarray)
        self.assertIsInstance(consensus, cp.ndarray)
        # La diagonal es 1, y son proporciones válidas
        self.assertTrue(cp.allclose(cp.diag(coincidence), 1.0))
        self.assertTrue(cp.all((coincidence >= 0) & (coincidence <= 1)))
        self.assertTrue(cp.all((consensus >= 0) & (consensus <= 1)))
        # La suma de cada entrada más el consensus da 1
        self.assertTrue(cp.allclose(coincidence + consensus, 1.0))

    def test_empty_matrix(self):
        with self.assertRaises(RuntimeError):
            ConsensusMatrixCupy(cp.empty((0, 4)))

    def test_insufficient_columns(self):
        with self.assertRaises(RuntimeError):
            ConsensusMatrixCupy(cp.zeros((3, 1)))

    def test_perfect_consensus(self):
        mat = cp.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        coincidence, consensus = ConsensusMatrixCupy(mat)
        self.assertTrue(cp.allclose(coincidence, cp.ones((3, 3))))
        self.assertTrue(cp.allclose(consensus, cp.zeros((3, 3))))

    def test_no_shared_clusters(self):
        mat = cp.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ])
        coincidence, consensus = ConsensusMatrixCupy(mat)
        self.assertTrue(cp.allclose(coincidence, cp.eye(3)))

if __name__ == "__main__":
    unittest.main()
