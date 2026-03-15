"""
Unit tests for JaccardValues utilities.
"""

######### Libraries #########
import unittest
import numpy as np
import pandas as pd

from ParetoInsight_CPU.JaccardValues import (
    jaccard_index_solutions,
    jaccard_index_clusters,
    compare_solutions_pair,
    find_equivalent_clusters_jaccard,
)


class TestJaccardValues(unittest.TestCase):
    """Test suite for Jaccard similarity utilities."""

    ##########################
    # Test Initialization
    ##########################

    def setUp(self):
        """
        Build reusable clustering solutions.

        Matrix example:
        3 solutions, 4 genes
        """
        self.solutions_matrix = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 2, 1, 2],
        ])

        self.cluster_solutions = [
            [{0, 1}, {2, 3}],
            [{0, 1}, {2, 3}],
            [{0, 2}, {1, 3}],
        ]

    ##########################
    # jaccard_index_solutions
    ##########################

    def test_jaccard_matrix_shape(self):
        """Purpose: confirm output matrix is square and symmetric."""
        J = jaccard_index_solutions(self.solutions_matrix)

        self.assertEqual(J.shape, (3, 3))
        self.assertTrue(np.allclose(J, J.T))
        self.assertTrue(np.all(np.diag(J) == 1.0))

    def test_identical_solutions(self):
        """Purpose: identical solutions must have Jaccard similarity 1."""
        J = jaccard_index_solutions(self.solutions_matrix)

        self.assertEqual(J[0, 1], 1.0)

    def test_invalid_matrix_type(self):
        """Purpose: non-numpy input raises TypeError."""
        with self.assertRaises(TypeError):
            jaccard_index_solutions([[1, 2], [3, 4]])

    def test_invalid_matrix_dimension(self):
        """Purpose: 1D array raises ValueError."""
        with self.assertRaises(ValueError):
            jaccard_index_solutions(np.array([1, 2, 3]))

    def test_empty_matrix(self):
        """Purpose: empty matrix raises ValueError."""
        with self.assertRaises(ValueError):
            jaccard_index_solutions(np.empty((0, 4)))

    ##########################
    # jaccard_index_clusters
    ##########################

    def test_cluster_jaccard_matrix(self):
        """Purpose: confirm cluster comparison returns correct shape."""
        M = jaccard_index_clusters(
            self.cluster_solutions[0],
            self.cluster_solutions[2]
        )

        self.assertEqual(M.shape, (2, 2))
        self.assertTrue(np.all(M >= 0))
        self.assertTrue(np.all(M <= 1))

    def test_cluster_invalid_type(self):
        """Purpose: non-list input raises TypeError."""
        with self.assertRaises(TypeError):
            jaccard_index_clusters("invalid", self.cluster_solutions[0])

    def test_cluster_empty_solution(self):
        """Purpose: empty cluster list raises ValueError."""
        with self.assertRaises(ValueError):
            jaccard_index_clusters([], self.cluster_solutions[0])

    ##########################
    # compare_solutions_pair
    ##########################

    def test_compare_solutions_pair_output(self):
        """Purpose: confirm matching returns valid tuples."""
        matches = compare_solutions_pair(
            0, 2, self.cluster_solutions
        )

        self.assertTrue(isinstance(matches, list))
        self.assertTrue(all(len(t) == 3 for t in matches))
        self.assertTrue(all(0 <= t[2] <= 1 for t in matches))

    ##########################
    # find_equivalent_clusters_jaccard
    ##########################

    def test_find_equivalent_clusters_dataframe(self):
        """Purpose: confirm DataFrame structure is correct."""
        df = find_equivalent_clusters_jaccard(self.cluster_solutions)

        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertIn("Solution 1", df.columns)
        self.assertIn("Solution 2", df.columns)
        self.assertIn("Cluster 1", df.columns)
        self.assertIn("Cluster 2", df.columns)
        self.assertIn("Jaccard Similarity", df.columns)

    def test_find_equivalent_invalid_input(self):
        """Purpose: invalid input type raises TypeError."""
        with self.assertRaises(TypeError):
            find_equivalent_clusters_jaccard("invalid")


if __name__ == "__main__":
    unittest.main()
