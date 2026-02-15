"""
Unit tests for SolutionClusterMatrix module.

Purpose:
- Validate correct grouping behavior.
- Validate both output modes ("sets" and "indices").
- Validate parallel vs non-parallel consistency.
- Validate input validation.
- Ensure deterministic cluster ordering.
"""

import unittest
import numpy as np

from ParetoInsight_CPU.SolutionClusterMatrix import (
    solution_cluster_matrix
)


class TestSolutionClusterMatrix(unittest.TestCase):

    ############################
    # Test Initialization
    ############################
    def setUp(self):
        """
        Build small deterministic test matrix:

        2 solutions
        5 genes

        Solution 0:
            A A B B C
        Solution 1:
            X Y X Y Z
        """

        self.genes = ["G1", "G2", "G3", "G4", "G5"]

        self.matrix = np.array([
            [0, 0, 1, 1, 2],
            [5, 6, 5, 6, 7]
        ])

    ############################
    # Functional Tests - Sets Mode
    ############################

    def test_sets_mode_basic(self):
        """Ensure correct clustering in sets mode."""
        result = solution_cluster_matrix(
            self.matrix,
            self.genes,
            mode="sets",
            parallel=False
        )

        self.assertEqual(len(result), 2)

        # Solution 0
        sol0 = result[0]
        expected0 = [
            {"G1", "G2"},
            {"G3", "G4"},
            {"G5"}
        ]
        self.assertEqual(sol0, expected0)

        # Solution 1
        sol1 = result[1]
        expected1 = [
            {"G1", "G3"},
            {"G2", "G4"},
            {"G5"}
        ]
        self.assertEqual(sol1, expected1)

    ############################
    # Functional Tests - Indices Mode
    ############################

    def test_indices_mode_basic(self):
        """Ensure correct clustering in indices mode."""
        result = solution_cluster_matrix(
            self.matrix,
            self.genes,
            mode="indices",
            parallel=False
        )

        self.assertEqual(len(result), 2)

        sol0 = result[0]
        expected0 = [
            np.array([0, 1]),
            np.array([2, 3]),
            np.array([4])
        ]

        for r, e in zip(sol0, expected0):
            np.testing.assert_array_equal(r, e)

    ############################
    # Parallel Consistency Tests
    ############################

    def test_parallel_equals_nonparallel_sets(self):
        """Parallel and non-parallel should produce identical sets output."""
        serial = solution_cluster_matrix(
            self.matrix,
            self.genes,
            mode="sets",
            parallel=False
        )

        parallel = solution_cluster_matrix(
            self.matrix,
            self.genes,
            mode="sets",
            parallel=True,
            max_workers=2
        )

        self.assertEqual(serial, parallel)

    def test_parallel_equals_nonparallel_indices(self):
        """Parallel and non-parallel should produce identical indices output."""
        serial = solution_cluster_matrix(
            self.matrix,
            self.genes,
            mode="indices",
            parallel=False
        )

        parallel = solution_cluster_matrix(
            self.matrix,
            self.genes,
            mode="indices",
            parallel=True,
            max_workers=2
        )

        for sol_s, sol_p in zip(serial, parallel):
            for a, b in zip(sol_s, sol_p):
                np.testing.assert_array_equal(a, b)

    ############################
    # Validation Tests
    ############################

    def test_invalid_matrix_type(self):
        with self.assertRaises(TypeError):
            solution_cluster_matrix(
                matrix="not_array",
                genes=self.genes
            )

    def test_invalid_matrix_dim(self):
        with self.assertRaises(ValueError):
            solution_cluster_matrix(
                matrix=np.array([1, 2, 3]),
                genes=self.genes
            )

    def test_empty_matrix(self):
        with self.assertRaises(ValueError):
            solution_cluster_matrix(
                matrix=np.empty((0, 5)),
                genes=self.genes
            )

    def test_gene_length_mismatch(self):
        with self.assertRaises(ValueError):
            solution_cluster_matrix(
                matrix=self.matrix,
                genes=["G1", "G2"]
            )

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            solution_cluster_matrix(
                matrix=self.matrix,
                genes=self.genes,
                mode="invalid"
            )

    ############################
    # Deterministic Ordering Test
    ############################

    def test_cluster_order_is_sorted_by_label(self):
        """
        Ensure clusters are returned sorted by label value.
        """

        matrix = np.array([
            [10, 5, 10, 3]
        ])
        genes = ["A", "B", "C", "D"]

        result = solution_cluster_matrix(
            matrix,
            genes,
            mode="sets"
        )

        # Labels sorted: 3,5,10
        expected = [
            {"D"},        # label 3
            {"B"},        # label 5
            {"A", "C"}    # label 10
        ]

        self.assertEqual(result[0], expected)


if __name__ == "__main__":
    unittest.main()
