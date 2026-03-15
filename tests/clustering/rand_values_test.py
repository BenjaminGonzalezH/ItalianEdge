"""
Unit tests for RandValues utilities.

Purpose of this file:
- Validate normal and edge-case behavior of Rand Index utilities.
- Confirm matrix symmetry, diagonal correctness and value bounds.
- Validate Adjusted Rand Index (ARI).
- Test cluster-level comparison and summary DataFrame.
"""

######### Libraries #########
import unittest
import numpy as np
import pandas as pd

from ParetoInsight_CPU.RandValues import (
    rand_index_solutions,
    adjusted_rand_index_solutions,
    rand_index_clusters,
    adjusted_rand_index_clusters,
    compare_solutions_pair,
    find_equivalent_clusters_rand,
)


class TestRandValues(unittest.TestCase):
    """Test suite for Rand and Adjusted Rand utilities."""

    ##########################
    # Test Initialization
    ##########################

    def setUp(self):
        """
        Build reusable clustering structures.
        """

        # 3 solutions, 4 genes
        self.solutions_matrix = np.array([
            [1, 1, 2, 2],  # solution 0
            [1, 1, 2, 2],  # identical to 0
            [1, 2, 1, 2],  # different structure
        ])

        # Cluster representations
        self.cluster_solutions = [
            [{0, 1}, {2, 3}],
            [{0, 1}, {2, 3}],
            [{0, 2}, {1, 3}],
        ]

    ##########################
    # rand_index_solutions
    ##########################

    def test_rand_matrix_shape(self):
        """Confirm Rand matrix is square, symmetric, and diagonal is 1."""
        R = rand_index_solutions(self.solutions_matrix)

        self.assertEqual(R.shape, (3, 3))
        self.assertTrue(np.allclose(R, R.T))
        self.assertTrue(np.all(np.diag(R) == 1.0))

    def test_rand_identical_solutions(self):
        """Identical solutions must have Rand = 1."""
        R = rand_index_solutions(self.solutions_matrix)
        self.assertEqual(R[0, 1], 1.0)

    def test_rand_invalid_input_type(self):
        """Non-numpy input should raise TypeError."""
        with self.assertRaises(TypeError):
            rand_index_solutions([[1, 2], [3, 4]])

    def test_rand_invalid_dimension(self):
        """1D input should raise ValueError."""
        with self.assertRaises(ValueError):
            rand_index_solutions(np.array([1, 2, 3]))

    ##########################
    # adjusted_rand_index_solutions
    ##########################

    def test_ari_matrix_properties(self):
        """ARI matrix must be symmetric with diagonal 1."""
        A = adjusted_rand_index_solutions(self.solutions_matrix)

        self.assertEqual(A.shape, (3, 3))
        self.assertTrue(np.allclose(A, A.T))
        self.assertTrue(np.all(np.diag(A) == 1.0))

    def test_ari_identical_solutions(self):
        """Identical solutions must have ARI = 1."""
        A = adjusted_rand_index_solutions(self.solutions_matrix)
        self.assertEqual(A[0, 1], 1.0)

    def test_ari_value_range(self):
        """ARI values must lie in [-1, 1]."""
        A = adjusted_rand_index_solutions(self.solutions_matrix)
        self.assertTrue(np.all(A <= 1.0))
        self.assertTrue(np.all(A >= -1.0))

    ##########################
    # Cluster-level Rand
    ##########################

    def test_rand_clusters_matrix(self):
        """Cluster-level Rand matrix shape and bounds."""
        M = rand_index_clusters(
            self.cluster_solutions[0],
            self.cluster_solutions[2]
        )

        self.assertEqual(M.shape, (2, 2))
        self.assertTrue(np.all(M >= 0))
        self.assertTrue(np.all(M <= 1))

    def test_ari_clusters_matrix(self):
        """Cluster-level ARI matrix shape and bounds."""
        M = adjusted_rand_index_clusters(
            self.cluster_solutions[0],
            self.cluster_solutions[2]
        )

        self.assertEqual(M.shape, (2, 2))
        self.assertTrue(np.all(M <= 1.0))
        self.assertTrue(np.all(M >= -1.0))

    def test_cluster_invalid_type(self):
        """Invalid cluster structure should raise TypeError."""
        with self.assertRaises(TypeError):
            rand_index_clusters("invalid", self.cluster_solutions[0])

    ##########################
    # compare_solutions_pair
    ##########################

    def test_compare_solutions_pair_rand(self):
        """Greedy matching returns valid tuples (Rand)."""
        matches = compare_solutions_pair(
            0, 2, self.cluster_solutions, metric="rand"
        )

        self.assertTrue(isinstance(matches, list))
        self.assertTrue(all(len(t) == 3 for t in matches))
        self.assertTrue(all(0 <= t[2] <= 1 for t in matches))

    def test_compare_solutions_pair_ari(self):
        """Greedy matching returns valid tuples (ARI)."""
        matches = compare_solutions_pair(
            0, 2, self.cluster_solutions, metric="adjusted_rand"
        )

        self.assertTrue(isinstance(matches, list))
        self.assertTrue(all(-1 <= t[2] <= 1 for t in matches))

    def test_compare_invalid_metric(self):
        """Invalid metric name should raise ValueError."""
        with self.assertRaises(ValueError):
            compare_solutions_pair(
                0, 2, self.cluster_solutions, metric="invalid"
            )

    ##########################
    # find_equivalent_clusters_rand
    ##########################

    def test_find_equivalent_clusters_dataframe(self):
        """Summary DataFrame structure check."""
        df = find_equivalent_clusters_rand(
            self.cluster_solutions,
            metric="rand"
        )

        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertIn("Solution 1", df.columns)
        self.assertIn("Solution 2", df.columns)
        self.assertIn("Cluster 1", df.columns)
        self.assertIn("Cluster 2", df.columns)
        self.assertIn("Similarity", df.columns)
        self.assertIn("Metric", df.columns)

    def test_find_equivalent_invalid_input(self):
        """Invalid input type should raise TypeError."""
        with self.assertRaises(TypeError):
            find_equivalent_clusters_rand("invalid")


if __name__ == "__main__":
    unittest.main()
