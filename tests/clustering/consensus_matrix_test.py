"""Unit tests for ConsensusMatrix module.

These tests validate the correctness of the consensus matrix computation
and verify proper handling of invalid inputs.
"""

import unittest

import numpy as np

from biocluster.clustering.consensus_matrix import consensus_matrix


class TestConsensusMatrix(unittest.TestCase):

    def setUp(self):
        """Prepare small clustering matrices used across tests."""

        self.matrix = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1]])

        self.perfect_consensus = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])

        self.no_shared = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # ──────────────────────────────────────────────────────────────────────────
    # Core Behavior
    # ──────────────────────────────────────────────────────────────────────────

    def test_regular_behavior(self):
        """Verify correct matrix shapes and mathematical properties."""

        coincidence, consensus = consensus_matrix(self.matrix)

        self.assertIsInstance(coincidence, np.ndarray)
        self.assertIsInstance(consensus, np.ndarray)

        self.assertEqual(coincidence.shape, (4, 4))
        self.assertEqual(consensus.shape, (4, 4))

        # Diagonal must be 1
        self.assertTrue(np.allclose(np.diag(coincidence), 1.0))

        # Symmetry property
        self.assertTrue(np.allclose(coincidence, coincidence.T))

        # Values must remain in valid probability range
        self.assertTrue((coincidence >= 0).all())
        self.assertTrue((coincidence <= 1).all())

        self.assertTrue((consensus >= 0).all())
        self.assertTrue((consensus <= 1).all())

        # Mathematical relationship
        self.assertTrue(np.allclose(consensus, 1 - coincidence))

    # ──────────────────────────────────────────────────────────────────────────
    # Input Validation
    # ──────────────────────────────────────────────────────────────────────────

    def test_empty_matrix(self):
        """Empty matrices should raise ValueError."""
        with self.assertRaises(ValueError):
            consensus_matrix(np.empty((0, 4)))

    def test_insufficient_columns(self):
        """Matrices with fewer than two columns are invalid."""
        with self.assertRaises(ValueError):
            consensus_matrix(np.zeros((3, 1)))

    def test_non_numpy_input(self):
        """Non-NumPy inputs should raise TypeError."""
        with self.assertRaises(TypeError):
            consensus_matrix([[1, 2], [3, 4]])

    def test_non_2d_input(self):
        """Non-2D arrays should raise ValueError."""
        with self.assertRaises(ValueError):
            consensus_matrix(np.array([1, 2, 3]))

    # ──────────────────────────────────────────────────────────────────────────
    # Mathematical Behavior
    # ──────────────────────────────────────────────────────────────────────────

    def test_perfect_consensus(self):
        """Verify expected coincidence matrix for deterministic clustering."""

        coincidence, consensus = consensus_matrix(self.perfect_consensus)

        expected = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)

        self.assertTrue(np.allclose(coincidence, expected))
        self.assertTrue(np.allclose(consensus, 1 - expected))

    def test_no_shared_clusters(self):
        """If no clusters overlap, coincidence matrix should be identity."""

        coincidence, consensus = consensus_matrix(self.no_shared)

        self.assertTrue(np.allclose(coincidence, np.eye(3)))
        self.assertTrue(np.allclose(consensus, np.ones((3, 3)) - np.eye(3)))

    def test_symmetry_property(self):
        """Coincidence matrix must be symmetric."""

        coincidence, _ = consensus_matrix(self.matrix)

        self.assertTrue(np.allclose(coincidence, coincidence.T))


if __name__ == "__main__":
    unittest.main()
