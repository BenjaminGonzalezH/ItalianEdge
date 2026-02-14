######### Libraries #########
import unittest
import numpy as np

from ParetoInsight_CPU.ConsensusMatrix import consensus_matrix


class TestConsensusMatrix(unittest.TestCase):

    ##########################
    # Setup
    ##########################
    def setUp(self):
        self.matrix = np.array([
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1]
        ])

        self.perfect_consensus = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0]
        ])

        self.no_shared = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

    ##########################
    # Core Behavior
    ##########################
    def test_regular_behavior(self):
        coincidence, consensus = consensus_matrix(self.matrix)

        self.assertEqual(coincidence.shape, (4, 4))
        self.assertEqual(consensus.shape, (4, 4))

        # Diagonal must be 1
        self.assertTrue(np.allclose(np.diag(coincidence), 1.0))

        # Symmetry
        self.assertTrue(np.allclose(coincidence, coincidence.T))

        # Valid range
        self.assertTrue((coincidence >= 0).all() and (coincidence <= 1).all())
        self.assertTrue((consensus >= 0).all() and (consensus <= 1).all())

        # Mathematical identity
        self.assertTrue(np.allclose(consensus, 1 - coincidence))

    ##########################
    # Edge Cases
    ##########################
    def test_empty_matrix(self):
        with self.assertRaises(ValueError):
            consensus_matrix(np.empty((0, 4)))

    def test_insufficient_columns(self):
        with self.assertRaises(ValueError):
            consensus_matrix(np.zeros((3, 1)))

    def test_non_numpy_input(self):
        with self.assertRaises(TypeError):
            consensus_matrix([[1, 2], [3, 4]])

    def test_non_2d_input(self):
        with self.assertRaises(ValueError):
            consensus_matrix(np.array([1, 2, 3]))

    ##########################
    # Mathematical Properties
    ##########################
    def test_perfect_consensus(self):
        coincidence, _ = consensus_matrix(self.perfect_consensus)

        expected = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1]
        ], dtype=float)

        self.assertTrue(np.allclose(coincidence, expected))

    def test_no_shared_clusters(self):
        coincidence, _ = consensus_matrix(self.no_shared)
        self.assertTrue(np.allclose(coincidence, np.eye(3)))

    def test_symmetry_property(self):
        coincidence, _ = consensus_matrix(self.matrix)
        self.assertTrue(np.allclose(coincidence, coincidence.T))


if __name__ == "__main__":
    unittest.main()
