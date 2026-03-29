"""
RandValues Tests
----------------

Unit tests for clustering similarity utilities based on:

    • Rand Index (RI)
    • Adjusted Rand Index (ARI)

Test objectives
---------------
1. Validate structure and symmetry of similarity matrices.
2. Confirm expected behavior for identical clusterings.
3. Verify value bounds for RI and ARI.
4. Validate cluster-level comparison functions.
5. Test greedy cluster matching behavior.
6. Validate DataFrame summaries for cluster equivalence.

All tests use small deterministic datasets so results can
be manually verified if needed.
"""

######### Libraries #########

import unittest
import numpy as np
import pandas as pd

from gclusters_characterization.clustering.rand_values import (
    rand_index_solutions,
    adjusted_rand_index_solutions,
    rand_index_clusters,
    adjusted_rand_index_clusters,
    compare_solutions_pair,
    find_equivalent_clusters_rand,
)


class TestRandValues(unittest.TestCase):
    """
    Test suite validating Rand Index utilities.
    """

    ##########################
    # Test Initialization
    ##########################

    def setUp(self):
        """
        Create deterministic clustering examples.

        Structure
        ---------
        3 clustering solutions applied to 4 elements.
        """

        # Solution-level representation
        self.solutions_matrix = np.array([
            [1, 1, 2, 2],  # solution 0
            [1, 1, 2, 2],  # identical solution
            [1, 2, 1, 2],  # different clustering
        ])

        # Cluster-level representation
        self.cluster_solutions = [
            [{0, 1}, {2, 3}],
            [{0, 1}, {2, 3}],
            [{0, 2}, {1, 3}],
        ]

    ##########################
    # rand_index_solutions
    ##########################

    def test_rand_matrix_structure(self):
        """Rand matrix must be square, symmetric and have diagonal=1."""
        R = rand_index_solutions(self.solutions_matrix)

        self.assertEqual(R.shape, (3, 3))
        self.assertTrue(np.allclose(R, R.T))
        self.assertTrue(np.allclose(np.diag(R), 1))

    def test_rand_identical_solutions(self):
        """Identical clustering solutions must yield RI = 1."""
        R = rand_index_solutions(self.solutions_matrix)

        self.assertEqual(R[0, 1], 1.0)

    def test_rand_value_bounds(self):
        """Rand Index values must lie between 0 and 1."""
        R = rand_index_solutions(self.solutions_matrix)

        self.assertTrue(np.all(R >= 0))
        self.assertTrue(np.all(R <= 1))

    def test_rand_invalid_type(self):
        """Non-numpy inputs must raise TypeError."""
        with self.assertRaises(TypeError):
            rand_index_solutions([[1, 2], [3, 4]])

    def test_rand_invalid_dimension(self):
        """1D input arrays must raise ValueError."""
        with self.assertRaises(ValueError):
            rand_index_solutions(np.array([1, 2, 3]))

    ##########################
    # adjusted_rand_index_solutions
    ##########################

    def test_ari_matrix_structure(self):
        """ARI matrix must be symmetric with diagonal equal to 1."""
        A = adjusted_rand_index_solutions(self.solutions_matrix)

        self.assertEqual(A.shape, (3, 3))
        self.assertTrue(np.allclose(A, A.T))
        self.assertTrue(np.allclose(np.diag(A), 1))

    def test_ari_identical_solutions(self):
        """Identical clusterings must produce ARI = 1."""
        A = adjusted_rand_index_solutions(self.solutions_matrix)

        self.assertEqual(A[0, 1], 1.0)

    def test_ari_value_range(self):
        """ARI values must lie between -1 and 1."""
        A = adjusted_rand_index_solutions(self.solutions_matrix)

        self.assertTrue(np.all(A <= 1))
        self.assertTrue(np.all(A >= -1))

    ##########################
    # Cluster-level Rand
    ##########################

    def test_rand_clusters_matrix(self):
        """Cluster-level Rand matrix must have expected shape."""
        M = rand_index_clusters(
            self.cluster_solutions[0],
            self.cluster_solutions[2],
        )

        self.assertEqual(M.shape, (2, 2))
        self.assertTrue(np.all(M >= 0))
        self.assertTrue(np.all(M <= 1))

    def test_rand_clusters_identical(self):
        """Identical cluster sets should produce RI = 1."""
        M = rand_index_clusters(
            self.cluster_solutions[0],
            self.cluster_solutions[1],
        )

        self.assertTrue(np.diag(M, 1))

    def test_cluster_invalid_structure(self):
        """Invalid cluster container must raise TypeError."""
        with self.assertRaises(TypeError):
            rand_index_clusters("invalid", self.cluster_solutions[0])

    ##########################
    # Cluster-level ARI
    ##########################

    def test_ari_clusters_matrix(self):
        """Cluster-level ARI matrix must be within valid bounds."""
        M = adjusted_rand_index_clusters(
            self.cluster_solutions[0],
            self.cluster_solutions[2],
        )

        self.assertEqual(M.shape, (2, 2))
        self.assertTrue(np.all(M <= 1))
        self.assertTrue(np.all(M >= -1))

    ##########################
    # compare_solutions_pair
    ##########################

    def test_compare_solutions_pair_rand(self):
        """Greedy matching must return cluster index pairs."""
        matches = compare_solutions_pair(
            0, 2, self.cluster_solutions, metric="rand"
        )

        self.assertTrue(isinstance(matches, list))
        self.assertTrue(all(len(m) == 3 for m in matches))
        self.assertTrue(all(0 <= m[2] <= 1 for m in matches))

    def test_compare_solutions_pair_ari(self):
        """Greedy matching must return valid ARI scores."""
        matches = compare_solutions_pair(
            0, 2, self.cluster_solutions, metric="adjusted_rand"
        )

        self.assertTrue(isinstance(matches, list))
        self.assertTrue(all(-1 <= m[2] <= 1 for m in matches))

    def test_compare_invalid_metric(self):
        """Invalid metric names must raise ValueError."""
        with self.assertRaises(ValueError):
            compare_solutions_pair(
                0, 2, self.cluster_solutions, metric="invalid"
            )

    ##########################
    # find_equivalent_clusters_rand
    ##########################

    def test_find_equivalent_clusters_dataframe(self):
        """Summary DataFrame must contain expected columns."""
        df = find_equivalent_clusters_rand(
            self.cluster_solutions,
            metric="rand"
        )

        self.assertTrue(isinstance(df, pd.DataFrame))

        expected_columns = {
            "Solution 1",
            "Solution 2",
            "Cluster 1",
            "Cluster 2",
            "rand Similarity",
        }

        self.assertTrue(expected_columns.issubset(df.columns))

    def test_find_equivalent_invalid_input(self):
        """Invalid input types must raise TypeError."""
        with self.assertRaises(TypeError):
            find_equivalent_clusters_rand("invalid")

    ##########################
    # Regression: cluster RI vs. sklearn
    ##########################

    def test_cluster_ri_matches_binary_sklearn(self):
        """
        Cluster-level RI must equal sklearn rand_score on binary
        membership vectors built over the cluster-pair union.

        Counter-example that caught the previous wrong formula:
          A0 = {0, 1}, B0 = {0, 2}  →  union = {0, 1, 2}
          a_vec = [1, 1, 0], b_vec = [1, 0, 1]
          Correct RI = 1/3  (one agreement out of three pairs)
        """
        from sklearn.metrics import rand_score as sklearn_rand_score

        # Cluster pair: A0={0,1} vs B0={0,2}; universe={0,1,2}
        a_vec = np.array([1, 1, 0])
        b_vec = np.array([1, 0, 1])
        expected = sklearn_rand_score(a_vec, b_vec)

        M = rand_index_clusters([{0, 1}, {2, 3}], [{0, 2}, {1, 3}])
        actual = float(M[0, 0])

        self.assertAlmostEqual(
            actual, expected, places=9,
            msg=f"Cluster RI {actual:.9f} != sklearn {expected:.9f}",
        )


if __name__ == "__main__":
    unittest.main()