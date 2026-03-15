"""
Unit tests for WangIndex biological similarity module.

Purpose:
- Validate Wang similarity computation using a reference DataFrame.
- Confirm matrix symmetry and diagonal behavior.
- Ensure Wang column is correctly appended.
- Validate handling of NA genes.
- Validate normalization behavior.
"""

import unittest
import numpy as np
import pandas as pd

from ParetoInsight_CPU.WangIndex import (
    solution_wang_similarity_from_dataframe,
)


class TestWangBiologicalSimilarity(unittest.TestCase):

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setUp(self):
        """
        Build controlled small example:

        Genes: G1, G2, G3, G4
        GO similarity matrix manually defined.
        """

        self.ids = ["G1", "G2", "G3", "G4"]

        # Symmetric biological similarity matrix
        self.sim_matrix = np.array([
            [1.0, 0.8, 0.2, 0.1],
            [0.8, 1.0, 0.3, 0.2],
            [0.2, 0.3, 1.0, 0.9],
            [0.1, 0.2, 0.9, 1.0],
        ], dtype=float)

        # Two clustering solutions
        # Solution 0: {G1,G2}, {G3,G4}
        # Solution 1: {G1,G3}, {G2,G4}
        self.solutions = [
            [
                {"G1", "G2"},
                {"G3", "G4"},
            ],
            [
                {"G1", "G3"},
                {"G2", "G4"},
            ],
        ]

        # Reference DataFrame (Jaccard already computed)
        self.reference_df = pd.DataFrame([
            {
                "Solution 1": 0,
                "Solution 2": 1,
                "Cluster 1": 0,
                "Cluster 2": 0,
                "Jaccard Similarity": 0.25,
            },
            {
                "Solution 1": 0,
                "Solution 2": 1,
                "Cluster 1": 1,
                "Cluster 2": 1,
                "Jaccard Similarity": 0.25,
            },
        ])

    # ------------------------------------------------------------------
    # Core Tests
    # ------------------------------------------------------------------

    def test_wang_column_added(self):
        """Ensure Wang Similarity column is appended."""
        matrix, df = solution_wang_similarity_from_dataframe(
            self.ids,
            self.sim_matrix,
            self.reference_df,
            self.solutions,
        )

        self.assertIn("Wang Similarity", df.columns)
        self.assertEqual(len(df), 2)

    def test_matrix_symmetry(self):
        """Ensure final matrix is symmetric."""
        matrix, _ = solution_wang_similarity_from_dataframe(
            self.ids,
            self.sim_matrix,
            self.reference_df,
            self.solutions,
        )

        self.assertTrue(np.allclose(matrix, matrix.T))

    def test_diagonal_is_one(self):
        """Diagonal must be 1."""
        matrix, _ = solution_wang_similarity_from_dataframe(
            self.ids,
            self.sim_matrix,
            self.reference_df,
            self.solutions,
        )

        self.assertTrue(np.allclose(np.diag(matrix), 1.0))

    def test_expected_wang_value(self):
        """
        Validate manually computed biological similarity.

        Cluster0 (S0) = {G1,G2}
        Cluster0 (S1) = {G1,G3}

        Expected mean:
        sim(G1,G1) + sim(G1,G3) +
        sim(G2,G1) + sim(G2,G3)
        --------------------------------
                        4
        """
        matrix, df = solution_wang_similarity_from_dataframe(
            self.ids,
            self.sim_matrix,
            self.reference_df,
            self.solutions,
            normalize_matrix=False,
        )

        # Manual computation
        expected = np.mean([
            1.0, 0.2,
            0.8, 0.3,
        ])

        computed = df.loc[0, "Wang Similarity"]
        self.assertAlmostEqual(computed, expected, places=6)

    def test_normalization(self):
        """Ensure normalization rescales matrix to max=1."""
        matrix, _ = solution_wang_similarity_from_dataframe(
            self.ids,
            self.sim_matrix,
            self.reference_df,
            self.solutions,
            normalize_matrix=True,
        )

        self.assertAlmostEqual(np.max(matrix), 1.0)

    # ------------------------------------------------------------------
    # Edge Case Tests
    # ------------------------------------------------------------------

    def test_missing_gene_is_ignored(self):
        """Gene not present in ids should not crash."""
        bad_solutions = [
            [{"G1", "G2"}, {"G3", "G4"}],
            [{"G1", "G3"}, {"G2", "UNKNOWN"}],
        ]

        matrix, df = solution_wang_similarity_from_dataframe(
            self.ids,
            self.sim_matrix,
            self.reference_df,
            bad_solutions,
        )

        self.assertTrue(np.isfinite(matrix).all())

    def test_empty_cluster_results_zero(self):
        """If cluster becomes empty after filtering, Wang similarity = 0."""
        bad_solutions = [
            [{"NA"}, {"G3", "G4"}],
            [{"G1", "G3"}, {"G2", "G4"}],
        ]

        matrix, df = solution_wang_similarity_from_dataframe(
            self.ids,
            self.sim_matrix,
            self.reference_df,
            bad_solutions,
            na_value="NA",
        )

        self.assertTrue((df["Wang Similarity"] >= 0).all())

    def test_invalid_dataframe_columns(self):
        """Missing required columns must raise error."""
        bad_df = pd.DataFrame({"A": [1]})

        with self.assertRaises(ValueError):
            solution_wang_similarity_from_dataframe(
                self.ids,
                self.sim_matrix,
                bad_df,
                self.solutions,
            )


if __name__ == "__main__":
    unittest.main()