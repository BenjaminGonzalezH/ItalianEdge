######### Libraries #########
import unittest                     # Test interface.
import numpy as np                  # Numbers ADT managment.
import sys                          # syscalls.
import io                           # Input-Output 
from ParetoInsight_CPU.ConsensusMatrix import (
    ConsensusMatrix
)

class TestConsensusMatrix(unittest.TestCase):

    ########################## Test's Initialization ##########################
    def setUp(self):
        self.matrix = np.array([
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1]
        ])

        self.perfect_consenmsus = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0]
        ])

        self.no_shared = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    ########################## Tests ##########################
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
        coincidence, _ = ConsensusMatrix(self.perfect_consenmsus)
        self.assertTrue(np.all(coincidence == np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1]
        ])))

    def test_no_shared_clusters(self):
        coincidence, _ = ConsensusMatrix(self.no_shared)
        self.assertTrue(np.allclose(coincidence, np.eye(3)))

if __name__ == "__main__":
    unittest.main()
