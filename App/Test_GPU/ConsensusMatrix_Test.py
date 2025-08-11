######### Libraries #########
import unittest                     # Test interface.
import numpy as np                  # Numbers ADT managment.
import cupy as cp                   # CPU math structures.
import sys                          # syscalls.
import io                           # Input-Output 
from ParetoInsight_GPU.ConsensusMatrix import (
    ConsensusMatrixCupy
)

class TestConsensusMatrixCupy(unittest.TestCase):
    def setUp(self):
        self.cpu_mat = np.array([
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1]
        ])

        self.perfect = cp.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])


        self.no_shared = cp.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ])

        self.gpu_mat = cp.array(self.cpu_mat)

        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def test_regular_behavior(self):
        coincidence, consensus = ConsensusMatrixCupy(self.gpu_mat)
        self.assertEqual(coincidence.shape, (4,4))
        self.assertIsInstance(coincidence, cp.ndarray)
        self.assertIsInstance(consensus, cp.ndarray)
        self.assertTrue(cp.allclose(cp.diag(coincidence), 1.0))
        self.assertTrue(cp.all((coincidence >= 0) & (coincidence <= 1)))
        self.assertTrue(cp.all((consensus >= 0) & (consensus <= 1)))
        self.assertTrue(cp.allclose(coincidence + consensus, 1.0))

    def test_empty_matrix(self):
        with self.assertRaises(RuntimeError):
            ConsensusMatrixCupy(cp.empty((0, 4)))

    def test_insufficient_columns(self):
        with self.assertRaises(RuntimeError):
            ConsensusMatrixCupy(cp.zeros((3, 1)))

    def test_perfect_consensus(self):
        coincidence, consensus = ConsensusMatrixCupy(self.perfect)
        self.assertTrue(cp.allclose(coincidence, cp.ones((3, 3))))
        self.assertTrue(cp.allclose(consensus, cp.zeros((3, 3))))

    def test_no_shared_clusters(self):
        coincidence, _ = ConsensusMatrixCupy(self.no_shared)
        self.assertTrue(cp.allclose(coincidence, cp.eye(3)))

if __name__ == "__main__":
    unittest.main()
