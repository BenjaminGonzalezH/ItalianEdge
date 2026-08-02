"""
Unit tests for solutioncluster_matrix module.

Purpose
-------
Validate clustering conversion utilities including:

• cluster grouping correctness
• output modes ("sets" and "indices")
• parallel vs serial consistency
• deterministic cluster ordering
• input validation

Tests use small deterministic datasets to ensure
replicability and ease of manual verification.
"""

import unittest

import numpy as np

from biocluster.clustering.solutioncluster_matrix import solution_cluster_matrix


class TestSolutionClusterMatrix(unittest.TestCase):

    def setUp(self):
        """
        Create deterministic clustering test matrix.

        Matrix:
            2 solutions
            5 genes
        """

        self.genes = ["G1", "G2", "G3", "G4", "G5"]

        self.matrix = np.array([[0, 0, 1, 1, 2], [5, 6, 5, 6, 7]])

    # --------------------------------------------------
    # Functional tests
    # --------------------------------------------------

    def test_sets_mode_basic(self):

        result = solution_cluster_matrix(self.matrix, self.genes, mode="sets")

        expected_solution0 = [{"G1", "G2"}, {"G3", "G4"}, {"G5"}]

        self.assertEqual(result[0], expected_solution0)

    def test_indices_mode_basic(self):

        result = solution_cluster_matrix(self.matrix, self.genes, mode="indices")

        expected = [np.array([0, 1]), np.array([2, 3]), np.array([4])]

        for r, e in zip(result[0], expected):
            np.testing.assert_array_equal(r, e)

    # --------------------------------------------------
    # Parallel consistency
    # --------------------------------------------------

    def test_parallel_equals_serial_sets(self):

        serial = solution_cluster_matrix(self.matrix, self.genes, mode="sets")

        parallel = solution_cluster_matrix(
            self.matrix, self.genes, mode="sets", parallel=True, max_workers=2
        )

        self.assertEqual(serial, parallel)

    def test_parallel_equals_serial_indices(self):

        serial = solution_cluster_matrix(self.matrix, self.genes, mode="indices")

        parallel = solution_cluster_matrix(
            self.matrix, self.genes, mode="indices", parallel=True, max_workers=2
        )

        for s, p in zip(serial, parallel):
            for a, b in zip(s, p):
                np.testing.assert_array_equal(a, b)

    # --------------------------------------------------
    # Ordering test
    # --------------------------------------------------

    def test_cluster_order_sorted_by_label(self):

        matrix = np.array([[10, 5, 10, 3]])
        genes = ["A", "B", "C", "D"]

        result = solution_cluster_matrix(matrix, genes)

        expected = [{"D"}, {"B"}, {"A", "C"}]

        self.assertEqual(result[0], expected)

    # --------------------------------------------------
    # Validation tests
    # --------------------------------------------------

    def test_invalid_matrix_type(self):

        with self.assertRaises(TypeError):
            solution_cluster_matrix("bad", self.genes)

    def test_invalid_matrix_dimension(self):

        with self.assertRaises(ValueError):
            solution_cluster_matrix(np.array([1, 2, 3]), self.genes)

    def test_empty_matrix(self):

        with self.assertRaises(ValueError):
            solution_cluster_matrix(np.empty((0, 5)), self.genes)

    def test_gene_length_mismatch(self):

        with self.assertRaises(ValueError):
            solution_cluster_matrix(self.matrix, ["A", "B"])

    def test_invalid_mode(self):

        with self.assertRaises(ValueError):
            solution_cluster_matrix(self.matrix, self.genes, mode="bad")


if __name__ == "__main__":
    unittest.main()
