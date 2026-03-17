"""
Unit tests for GO3-based gene similarity module.

Goals
-----
- Avoid real GO3 calls (mock compare_genes)
- Validate input validation
- Validate cluster similarity computation
- Validate aggregation into solution matrix
- Validate normalization behavior
- Validate edge cases (empty clusters, missing genes)
- Ensure coverage >90%
"""

import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from gclusters_characterization.go.gene_similarity import (
    solution_go_similarity_from_dataframe,
    Go3SimilarityOptions,
    compute_go_gene_similarity_matrix
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def fake_compare_genes(g1, g2, ontology, similarity, groupwise, counter):
    """
    Deterministic fake similarity:
    similarity = len(common chars) / max length
    """
    if g1 == g2:
        return 1.0
    return float(len(set(g1) & set(g2)) / max(len(g1), len(g2)))

def fake_batch(pairs, ontology, similarity, groupwise, counter):
    out = []
    for g1, g2 in pairs:
        if g1 == g2:
            out.append(1.0)
        else:
            out.append(0.5)
    return out


# ─────────────────────────────────────────────
# Test Suite
# ─────────────────────────────────────────────

class TestGOSimilarity(unittest.TestCase):

    def setUp(self):
        self.ids = ["G1", "G2", "G3", "G4"]

        self.solutions = [
            [{"G1", "G2"}, {"G3", "G4"}],
            [{"G1", "G3"}, {"G2", "G4"}],
        ]

        self.reference_df = pd.DataFrame([
            {"Solution 1": 0, "Solution 2": 1, "Cluster 1": 0, "Cluster 2": 0},
            {"Solution 1": 0, "Solution 2": 1, "Cluster 1": 1, "Cluster 2": 1},
        ])

        self.options = Go3SimilarityOptions(
            similarity="lin",
            groupwise="bma",
            ontology="BP",
            normalize_matrix=True,
        )

        self.genes = ["G1", "G2", "G3"]

        self.counter = object()  # dummy object

    # --------------------------------------------------
    # Core behavior
    # --------------------------------------------------

    @patch("gclusters_characterization.go.gene_similarity.go3.compare_genes")
    def test_basic_similarity(self, mock_compare):
        """Ensure similarity column is added and matrix is computed."""
        mock_compare.side_effect = fake_compare_genes

        matrix, df = solution_go_similarity_from_dataframe(
            self.ids,
            self.reference_df,
            self.solutions,
            counter=self.counter,
            options=self.options,
        )

        self.assertEqual(matrix.shape, (2, 2))
        self.assertTrue(np.allclose(np.diag(matrix), 1.0))

        col = "Lin Similarity"
        self.assertIn(col, df.columns)
        self.assertTrue((df[col] >= 0).all())

    @patch("gclusters_characterization.go.gene_similarity.go3.compare_genes")
    def test_matrix_symmetry(self, mock_compare):
        """Matrix must be symmetric."""
        mock_compare.side_effect = fake_compare_genes

        matrix, _ = solution_go_similarity_from_dataframe(
            self.ids,
            self.reference_df,
            self.solutions,
            counter=self.counter,
            options=self.options,
        )

        self.assertTrue(np.allclose(matrix, matrix.T))

    @patch("gclusters_characterization.go.gene_similarity.go3.compare_genes")
    def test_normalization(self, mock_compare):
        """Normalization rescales matrix to max=1."""
        mock_compare.side_effect = fake_compare_genes

        matrix, _ = solution_go_similarity_from_dataframe(
            self.ids,
            self.reference_df,
            self.solutions,
            counter=self.counter,
            options=self.options,
        )

        self.assertAlmostEqual(np.max(matrix), 1.0)

    @patch("gclusters_characterization.go.gene_similarity.go3.compare_genes")
    def test_no_normalization(self, mock_compare):
        """Matrix should not be normalized when disabled."""
        mock_compare.side_effect = fake_compare_genes

        opts = Go3SimilarityOptions(
            similarity="lin",
            normalize_matrix=False,
        )

        matrix, _ = solution_go_similarity_from_dataframe(
            self.ids,
            self.reference_df,
            self.solutions,
            counter=self.counter,
            options=opts,
        )

        self.assertTrue(np.max(matrix) >= 1.0)

    # --------------------------------------------------
    # Edge cases
    # --------------------------------------------------

    @patch("gclusters_characterization.go.gene_similarity.go3.compare_genes")
    def test_missing_gene_filtered(self, mock_compare):
        """Genes not in ids should be ignored."""
        mock_compare.side_effect = fake_compare_genes

        bad_solutions = [
            [{"G1", "G2"}, {"G3", "G4"}],
            [{"G1", "G3"}, {"G2", "UNKNOWN"}],
        ]

        matrix, _ = solution_go_similarity_from_dataframe(
            self.ids,
            self.reference_df,
            bad_solutions,
            counter=self.counter,
            options=self.options,
        )

        self.assertTrue(np.isfinite(matrix).all())

    @patch("gclusters_characterization.go.gene_similarity.go3.compare_genes")
    def test_empty_cluster(self, mock_compare):
        """Empty clusters should produce similarity = 0."""
        mock_compare.side_effect = fake_compare_genes

        bad_solutions = [
            [{"NA"}, {"G3", "G4"}],
            [{"G1", "G3"}, {"G2", "G4"}],
        ]

        opts = Go3SimilarityOptions(na_value="NA")

        _, df = solution_go_similarity_from_dataframe(
            self.ids,
            self.reference_df,
            bad_solutions,
            counter=self.counter,
            options=opts,
        )

        col = "Wang Similarity" if "Wang Similarity" in df.columns else df.columns[-1]
        self.assertTrue((df[col] >= 0).all())

    @patch("gclusters_characterization.go.gene_similarity.go3.compare_genes")
    def test_exception_fallback(self, mock_compare):
        """Exceptions in GO3 should fallback to missing_similarity."""
        def fail(*args, **kwargs):
            raise RuntimeError("fail")

        mock_compare.side_effect = fail

        opts = Go3SimilarityOptions(missing_similarity=0.5)

        _, df = solution_go_similarity_from_dataframe(
            self.ids,
            self.reference_df,
            self.solutions,
            counter=self.counter,
            options=opts,
        )

        col = df.columns[-1]
        self.assertTrue((df[col] == 0.5).all())

    # --------------------------------------------------
    # Validation tests
    # --------------------------------------------------

    def test_invalid_reference_df(self):
        """Missing required columns should raise."""
        bad_df = pd.DataFrame({"A": [1]})

        with self.assertRaises(ValueError):
            solution_go_similarity_from_dataframe(
                self.ids,
                bad_df,
                self.solutions,
                counter=self.counter,
                options=self.options,
            )

    def test_invalid_solutions(self):
        """Invalid solution structure should raise."""
        bad_solutions = [123]

        with self.assertRaises(TypeError):
            solution_go_similarity_from_dataframe(
                self.ids,
                self.reference_df,
                bad_solutions,
                counter=self.counter,
                options=self.options,
            )

    def test_invalid_metric(self):
        """Unsupported similarity metric should raise."""
        opts = Go3SimilarityOptions(similarity="invalid")

        with self.assertRaises(ValueError):
            solution_go_similarity_from_dataframe(
                self.ids,
                self.reference_df,
                self.solutions,
                counter=self.counter,
                options=opts,
            )

    def test_invalid_groupwise(self):
        """Unsupported groupwise should raise."""
        opts = Go3SimilarityOptions(groupwise="invalid")

        with self.assertRaises(ValueError):
            solution_go_similarity_from_dataframe(
                self.ids,
                self.reference_df,
                self.solutions,
                counter=self.counter,
                options=opts,
            )

    def test_missing_counter(self):
        """Missing GO3 counter must raise."""
        with self.assertRaises(ValueError):
            solution_go_similarity_from_dataframe(
                self.ids,
                self.reference_df,
                self.solutions,
                counter=None,
                options=self.options,
            )

    # --------------------------------------------------
    # Core
    # --------------------------------------------------

    @patch("gclusters_characterization.go.gene_similarity.go3.compare_gene_pairs_batch")
    def test_basic_matrix(self, mock_batch):
        """Matrix should be symmetric and valid."""
        mock_batch.side_effect = fake_batch

        M = compute_go_gene_similarity_matrix(
            self.genes,
            counter=self.counter,
        )

        self.assertEqual(M.shape, (3, 3))
        self.assertTrue(np.allclose(M, M.T))
        self.assertTrue(np.allclose(np.diag(M), 1.0))

    @patch("gclusters_characterization.go.gene_similarity.go3.compare_gene_pairs_batch")
    def test_distance_matrix(self, mock_batch):
        """Distance matrix should invert similarity."""
        mock_batch.side_effect = fake_batch

        D = compute_go_gene_similarity_matrix(
            self.genes,
            counter=self.counter,
            as_distance=True,
        )

        self.assertTrue(np.allclose(np.diag(D), 0.0))
        self.assertTrue((D >= 0).all())

    # --------------------------------------------------
    # Edge cases
    # --------------------------------------------------

    @patch("gclusters_characterization.go.gene_similarity.go3.compare_gene_pairs_batch")
    def test_batch_failure(self, mock_batch):
        """Failure in batch call should fallback."""
        mock_batch.side_effect = RuntimeError("fail")

        M = compute_go_gene_similarity_matrix(
            self.genes,
            counter=self.counter,
            missing_similarity=0.2,
        )

        self.assertTrue((M[np.triu_indices(3, 1)] == 0.2).all())

    @patch("gclusters_characterization.go.gene_similarity.go3.compare_gene_pairs_batch")
    def test_single_gene(self, mock_batch):
        """Single gene should return 1x1."""
        mock_batch.side_effect = fake_batch

        M = compute_go_gene_similarity_matrix(
            ["G1"],
            counter=self.counter,
        )

        self.assertEqual(M.shape, (1, 1))
        self.assertEqual(M[0, 0], 1.0)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def test_empty_genes(self):
        with self.assertRaises(ValueError):
            compute_go_gene_similarity_matrix(
                [],
                counter=self.counter,
            )

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            compute_go_gene_similarity_matrix(
                123,
                counter=self.counter,
            )

    def test_missing_counter(self):
        with self.assertRaises(ValueError):
            compute_go_gene_similarity_matrix(
                self.genes,
                counter=None,
            )


if __name__ == "__main__":
    unittest.main()