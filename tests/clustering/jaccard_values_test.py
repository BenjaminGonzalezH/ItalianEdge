"""
Unit tests for JaccardValues utilities.

This test suite validates the correctness and robustness of
Jaccard similarity utilities used for clustering comparison.

Test coverage includes:

- solution-level Jaccard similarity
- cluster-level Jaccard similarity
- greedy cluster matching
- equivalent cluster detection across solutions
- input validation
- degenerate cases
"""

#############################
# Libraries
#############################

import unittest
import numpy as np
import pandas as pd

from gclusters_characterization.clustering.jaccard_values import (
    jaccard_index_solutions,
    jaccard_index_clusters,
    compare_solutions_pair,
    find_equivalent_clusters_jaccard,
)


class TestJaccardValues(unittest.TestCase):
    """
    Test suite validating Jaccard-based clustering comparison utilities.
    """

    #################################################
    # Test initialization
    #################################################

    def setUp(self):
        """
        Construct small clustering examples.

        Matrix example:

        Solution 0 → [1,1,2,2]
        Solution 1 → identical to solution 0
        Solution 2 → cross grouping
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

    #################################################
    # Tests: jaccard_index_solutions
    #################################################

    def test_jaccard_matrix_shape(self):
        """
        Purpose
        -------
        Verify that the resulting similarity matrix
        is square and symmetric.
        """

        J = jaccard_index_solutions(self.solutions_matrix)

        self.assertEqual(J.shape, (3, 3))
        self.assertTrue(np.allclose(J, J.T))
        self.assertTrue(np.all(np.diag(J) == 1.0))

    def test_identical_solutions(self):
        """
        Purpose
        -------
        Identical clustering solutions must
        produce similarity equal to 1.
        """

        J = jaccard_index_solutions(self.solutions_matrix)

        self.assertEqual(J[0, 1], 1.0)

    def test_jaccard_value_range(self):
        """
        Purpose
        -------
        All similarity values must lie in [0,1].
        """

        J = jaccard_index_solutions(self.solutions_matrix)

        self.assertTrue(np.all(J >= 0))
        self.assertTrue(np.all(J <= 1))

    def test_invalid_matrix_type(self):
        """
        Purpose
        -------
        Non-numpy inputs must raise TypeError.
        """

        with self.assertRaises(TypeError):
            jaccard_index_solutions([[1, 2], [3, 4]])

    def test_invalid_matrix_dimension(self):
        """
        Purpose
        -------
        One-dimensional arrays must raise ValueError.
        """

        with self.assertRaises(ValueError):
            jaccard_index_solutions(np.array([1, 2, 3]))

    def test_empty_matrix(self):
        """
        Purpose
        -------
        Empty matrices must raise ValueError.
        """

        with self.assertRaises(ValueError):
            jaccard_index_solutions(np.empty((0, 4)))

    def test_denominator_zero_case(self):
        """
        Purpose
        -------
        Validate behavior when the Jaccard denominator is zero.

        This occurs when no gene pairs share cluster membership
        across solutions.
        """

        matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])

        J = jaccard_index_solutions(matrix)

        self.assertTrue(np.all(J >= 0))
        self.assertTrue(np.all(J <= 1))

    #################################################
    # Tests: jaccard_index_clusters
    #################################################

    def test_cluster_jaccard_matrix(self):
        """
        Purpose
        -------
        Verify that the cluster comparison matrix
        has the correct dimensions and valid values.
        """

        M = jaccard_index_clusters(
            self.cluster_solutions[0],
            self.cluster_solutions[2]
        )

        self.assertEqual(M.shape, (2, 2))
        self.assertTrue(np.all(M >= 0))
        self.assertTrue(np.all(M <= 1))

    def test_cluster_jaccard_identity(self):
        """
        Purpose
        -------
        Identical cluster solutions should produce
        perfect similarity along the diagonal.
        """

        M = jaccard_index_clusters(
            self.cluster_solutions[0],
            self.cluster_solutions[1]
        )

        self.assertEqual(M[0, 0], 1.0)
        self.assertEqual(M[1, 1], 1.0)

    def test_cluster_invalid_type(self):
        """
        Purpose
        -------
        Invalid input types must raise TypeError.
        """

        with self.assertRaises(TypeError):
            jaccard_index_clusters("invalid", self.cluster_solutions[0])

    def test_cluster_empty_solution(self):
        """
        Purpose
        -------
        Empty cluster lists must raise ValueError.
        """

        with self.assertRaises(ValueError):
            jaccard_index_clusters([], self.cluster_solutions[0])

    #################################################
    # Tests: compare_solutions_pair
    #################################################

    def test_compare_solutions_pair_output(self):
        """
        Purpose
        -------
        Verify that the function returns valid
        matching tuples.
        """

        matches = compare_solutions_pair(
            0,
            2,
            self.cluster_solutions
        )

        self.assertTrue(isinstance(matches, list))
        self.assertTrue(all(len(t) == 3 for t in matches))
        self.assertTrue(all(0 <= t[2] <= 1 for t in matches))

    def test_compare_solutions_pair_length(self):
        """
        Purpose
        -------
        Number of matches must not exceed
        the number of clusters in either solution.
        """

        matches = compare_solutions_pair(
            0,
            2,
            self.cluster_solutions
        )

        self.assertLessEqual(len(matches), 2)

    #################################################
    # Tests: find_equivalent_clusters_jaccard
    #################################################

    def test_find_equivalent_clusters_dataframe(self):
        """
        Purpose
        -------
        Ensure the resulting structure is a valid DataFrame
        with the expected columns.
        """

        df = find_equivalent_clusters_jaccard(self.cluster_solutions)

        self.assertTrue(isinstance(df, pd.DataFrame))

        expected_columns = [
            "Solution 1",
            "Solution 2",
            "Cluster 1",
            "Cluster 2",
            "Jaccard Similarity",
        ]

        for col in expected_columns:
            self.assertIn(col, df.columns)

    def test_find_equivalent_clusters_value_range(self):
        """
        Purpose
        -------
        Similarity values must be within [0,1].
        """

        df = find_equivalent_clusters_jaccard(self.cluster_solutions)

        self.assertTrue((df["Jaccard Similarity"] >= 0).all())
        self.assertTrue((df["Jaccard Similarity"] <= 1).all())

    def test_find_equivalent_invalid_input(self):
        """
        Purpose
        -------
        Invalid input type must raise TypeError.
        """

        with self.assertRaises(TypeError):
            find_equivalent_clusters_jaccard("invalid")


#################################################
# Test runner
#################################################

if __name__ == "__main__":
    unittest.main()