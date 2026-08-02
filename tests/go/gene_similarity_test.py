"""
gene_similarity_test.py

This module contains unit tests for GO enrichment, annotation utilities,
and gene similarity computation.

Functions tested:
1. go_enrichment:
   Validates enrichment computation, input handling, and scoring.

2. annotation_from_entrez_ids:
   Validates annotation retrieval and chunk processing.

3. _safe_neglog10:
   Validates numerical stability for logarithmic transformation.

4. compute_gene_similarity_matrix_by_batch:
   Validates batch-mode gene similarity matrix construction via GO3.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from biocluster.go.gene_similarity import (
    GeneSimilarityOptions,
    compute_gene_similarity_matrix_by_batch,
)
from biocluster.go.go_enrichment import (
    AnnotationOptions,
    GoEnrichmentOptions,
    _safe_neglog10,
    annotation_from_entrez_ids,
    go_enrichment,
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------


def fake_enrichment_df():
    """Create a minimal enrichment result for testing."""
    return pd.DataFrame(
        {
            "native": ["GO:0001", "GO:0002"],
            "p_value": [0.001, 0.01],
            "precision": [0.5, 0.25],
        }
    )


def fake_annotation_df(block):
    """Create a minimal annotation response for a gene block."""
    return pd.DataFrame(
        {
            "native": ["GO:0001", "GO:0002"],
            "intersections": [[block[0]], [block[-1]]],
        }
    )


# --------------------------------------------------
# Enrichment tests
# --------------------------------------------------


class TestGoEnrichment(unittest.TestCase):

    def setUp(self):
        """Prepare a small set of genes including duplicates and invalid values."""
        self.genes = ["1", "2", "3", "3", None, ""]

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_basic_enrichment(self, mock_gp):
        """Ensure enrichment runs and produces expected columns."""
        mock_instance = MagicMock()
        mock_instance.profile.return_value = fake_enrichment_df()
        mock_gp.return_value = mock_instance

        df = go_enrichment(self.genes)

        # Input cleaning should reduce duplicates
        self.assertEqual(mock_instance.profile.call_count, 1)

        # Expected computed columns
        self.assertIn("gene_ratio", df.columns)
        self.assertIn("qscore", df.columns)

        # Results should be sorted by p-value
        self.assertTrue(df["p_value"].is_monotonic_increasing)

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_missing_precision_column(self, mock_gp):
        """Enrichment should still work without 'precision'."""
        df_mock = fake_enrichment_df().drop(columns=["precision"])

        mock_instance = MagicMock()
        mock_instance.profile.return_value = df_mock
        mock_gp.return_value = mock_instance

        df = go_enrichment(["1", "2"])

        self.assertIn("qscore", df.columns)

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_missing_pvalue_column(self, mock_gp):
        """If p_value is missing, qscore should not be computed."""
        df_mock = fake_enrichment_df().drop(columns=["p_value"])

        mock_instance = MagicMock()
        mock_instance.profile.return_value = df_mock
        mock_gp.return_value = mock_instance

        df = go_enrichment(["1", "2"])

        self.assertNotIn("qscore", df.columns)

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_empty_response(self, mock_gp):
        """Empty API response should return an empty DataFrame."""
        mock_instance = MagicMock()
        mock_instance.profile.return_value = pd.DataFrame()
        mock_gp.return_value = mock_instance

        df = go_enrichment(["1"])
        self.assertTrue(df.empty)

    def test_invalid_inputs(self):
        """Invalid inputs should raise appropriate errors."""
        with self.assertRaises(ValueError):
            go_enrichment([])

        with self.assertRaises(TypeError):
            go_enrichment(123)

    def test_invalid_threshold(self):
        """Invalid threshold values should raise an error."""
        opts = GoEnrichmentOptions(user_threshold=0)
        with self.assertRaises(ValueError):
            go_enrichment(["1"], options=opts)

    def test_invalid_sources(self):
        """Empty source configuration should raise an error."""
        opts = GoEnrichmentOptions(sources=())
        with self.assertRaises(ValueError):
            go_enrichment(["1"], options=opts)

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_retry_logic(self, mock_gp):
        """Ensure retry mechanism is triggered when API fails."""
        mock_instance = MagicMock()

        calls = {"count": 0}

        def side_effect(**kwargs):
            calls["count"] += 1
            if calls["count"] < 2:
                raise RuntimeError("fail")
            return fake_enrichment_df()

        mock_instance.profile.side_effect = side_effect
        mock_gp.return_value = mock_instance

        df = go_enrichment(["1"])

        self.assertTrue(len(df) > 0)
        self.assertEqual(calls["count"], 2)


# --------------------------------------------------
# Annotation tests
# --------------------------------------------------


class TestAnnotation(unittest.TestCase):

    def setUp(self):
        """Prepare a small gene set for annotation tests."""
        self.genes = ["A", "B", "C", "D"]

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_basic_annotation(self, mock_gp):
        """Annotation should return a gene-to-terms mapping."""
        mock_instance = MagicMock()

        def side_effect(**kwargs):
            return fake_annotation_df(kwargs["query"])

        mock_instance.profile.side_effect = side_effect
        mock_gp.return_value = mock_instance

        result = annotation_from_entrez_ids(self.genes)

        self.assertIsInstance(result, dict)
        self.assertTrue(len(result) > 0)

        for k, v in result.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, list)

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_chunking(self, mock_gp):
        """Annotation should split queries according to chunk size."""
        mock_instance = MagicMock()

        calls = []

        def side_effect(**kwargs):
            calls.append(kwargs["query"])
            return fake_annotation_df(kwargs["query"])

        mock_instance.profile.side_effect = side_effect
        mock_gp.return_value = mock_instance

        opts = AnnotationOptions(chunk_size=2, n_threads=1)

        annotation_from_entrez_ids(self.genes, options=opts)

        self.assertEqual(len(calls), 2)

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_thread_failure_handling(self, mock_gp):
        """Failures in individual chunks should not crash execution."""
        mock_instance = MagicMock()

        def side_effect(**kwargs):
            if len(kwargs["query"]) == 2:
                raise RuntimeError("fail")
            return fake_annotation_df(kwargs["query"])

        mock_instance.profile.side_effect = side_effect
        mock_gp.return_value = mock_instance

        result = annotation_from_entrez_ids(self.genes)

        self.assertIsInstance(result, dict)

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_missing_columns(self, mock_gp):
        """Missing expected columns should return empty result."""
        mock_instance = MagicMock()
        mock_instance.profile.return_value = pd.DataFrame({"foo": [1]})
        mock_gp.return_value = mock_instance

        result = annotation_from_entrez_ids(self.genes)

        self.assertEqual(result, {})

    def test_invalid_inputs(self):
        """Invalid inputs should raise errors."""
        with self.assertRaises(ValueError):
            annotation_from_entrez_ids([])

        with self.assertRaises(TypeError):
            annotation_from_entrez_ids(123)


# --------------------------------------------------
# Utility tests
# --------------------------------------------------


class TestUtilities(unittest.TestCase):

    def test_safe_neglog10_normal(self):
        """Valid p-values should produce expected log scores."""
        val = _safe_neglog10(0.01)
        self.assertAlmostEqual(val, 2.0, places=5)

    def test_safe_neglog10_zero(self):
        """Zero p-value should return a finite value."""
        val = _safe_neglog10(0)
        self.assertTrue(np.isfinite(val))

    def test_safe_neglog10_invalid(self):
        """Invalid inputs should return NaN."""
        val = _safe_neglog10("bad")
        self.assertTrue(np.isnan(val))


# --------------------------------------------------
# Gene similarity by batch tests
# --------------------------------------------------


class TestComputeGeneSimilarityMatrixByBatch(unittest.TestCase):

    def setUp(self):
        self.genes = ["GeneA", "GeneB", "GeneC"]
        self.obo_path = "/fake/go.obo"
        self.gaf_path = "/fake/annotations.gaf"

    def _configure_mock(self, mock_go3, scores):
        mock_go3.load_gaf.return_value = MagicMock()
        mock_go3.build_term_counter.return_value = MagicMock()
        mock_go3.compare_gene_pairs_batch.return_value = scores

    @patch("biocluster.go.gene_similarity.go3")
    def test_return_types(self, mock_go3):
        """Function should return a list and an ndarray."""
        self._configure_mock(mock_go3, [0.8, 0.6, 0.7])

        genes_out, sim = compute_gene_similarity_matrix_by_batch(
            self.genes, obo_path=self.obo_path, gaf_path=self.gaf_path
        )

        self.assertIsInstance(genes_out, list)
        self.assertIsInstance(sim, np.ndarray)

    @patch("biocluster.go.gene_similarity.go3")
    def test_matrix_shape_and_gene_order(self, mock_go3):
        """Similarity matrix must be n×n and genes list must match input order."""
        n = len(self.genes)
        self._configure_mock(mock_go3, [0.5] * (n * (n - 1) // 2))

        genes_out, sim = compute_gene_similarity_matrix_by_batch(
            self.genes, obo_path=self.obo_path, gaf_path=self.gaf_path
        )

        self.assertEqual(sim.shape, (n, n))
        self.assertEqual(genes_out, self.genes)

    @patch("biocluster.go.gene_similarity.go3")
    def test_diagonal_is_one(self, mock_go3):
        """Diagonal entries must be 1.0 (self-similarity)."""
        self._configure_mock(mock_go3, [0.4, 0.5, 0.6])

        _, sim = compute_gene_similarity_matrix_by_batch(
            self.genes, obo_path=self.obo_path, gaf_path=self.gaf_path
        )

        np.testing.assert_array_equal(np.diag(sim), np.ones(len(self.genes)))

    @patch("biocluster.go.gene_similarity.go3")
    def test_matrix_symmetric(self, mock_go3):
        """Similarity matrix must be symmetric (sim[i,j] == sim[j,i])."""
        self._configure_mock(mock_go3, [0.3, 0.7, 0.9])

        _, sim = compute_gene_similarity_matrix_by_batch(
            self.genes, obo_path=self.obo_path, gaf_path=self.gaf_path
        )

        np.testing.assert_array_equal(sim, sim.T)

    @patch("biocluster.go.gene_similarity.go3")
    def test_scores_correctly_placed(self, mock_go3):
        """Batch scores must map to the correct (i,j) and (j,i) positions.

        pairs from combinations(["GeneA","GeneB","GeneC"], 2):
          (A,B) -> (0,1), (A,C) -> (0,2), (B,C) -> (1,2)
        """
        self._configure_mock(mock_go3, [0.1, 0.2, 0.3])

        _, sim = compute_gene_similarity_matrix_by_batch(
            self.genes, obo_path=self.obo_path, gaf_path=self.gaf_path
        )

        self.assertAlmostEqual(sim[0, 1], 0.1)
        self.assertAlmostEqual(sim[1, 0], 0.1)
        self.assertAlmostEqual(sim[0, 2], 0.2)
        self.assertAlmostEqual(sim[2, 0], 0.2)
        self.assertAlmostEqual(sim[1, 2], 0.3)
        self.assertAlmostEqual(sim[2, 1], 0.3)

    @patch("biocluster.go.gene_similarity.go3")
    def test_single_gene_returns_identity(self, mock_go3):
        """A single gene must yield a 1×1 matrix with value 1.0 and no pairs."""
        self._configure_mock(mock_go3, [])

        genes_out, sim = compute_gene_similarity_matrix_by_batch(
            ["GeneA"], obo_path=self.obo_path, gaf_path=self.gaf_path
        )

        self.assertEqual(genes_out, ["GeneA"])
        self.assertEqual(sim.shape, (1, 1))
        self.assertAlmostEqual(sim[0, 0], 1.0)

    @patch("biocluster.go.gene_similarity.go3")
    def test_load_go_terms_called_when_enabled(self, mock_go3):
        """go3.load_go_terms must be called with obo_path when load_go_terms=True."""
        self._configure_mock(mock_go3, [0.5])

        opts = GeneSimilarityOptions(load_go_terms=True)
        compute_gene_similarity_matrix_by_batch(
            ["G1", "G2"], obo_path=self.obo_path, gaf_path=self.gaf_path, go3_opts=opts
        )

        mock_go3.load_go_terms.assert_called_once_with(str(self.obo_path))

    @patch("biocluster.go.gene_similarity.go3")
    def test_load_go_terms_skipped_when_disabled(self, mock_go3):
        """go3.load_go_terms must NOT be called when load_go_terms=False."""
        self._configure_mock(mock_go3, [0.5])

        opts = GeneSimilarityOptions(load_go_terms=False)
        compute_gene_similarity_matrix_by_batch(
            ["G1", "G2"], obo_path=self.obo_path, gaf_path=self.gaf_path, go3_opts=opts
        )

        mock_go3.load_go_terms.assert_not_called()

    def test_invalid_empty_genes_raises(self):
        """Empty gene list must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_gene_similarity_matrix_by_batch(
                [], obo_path=self.obo_path, gaf_path=self.gaf_path
            )

    def test_invalid_genes_type_raises(self):
        """Non-list/tuple input must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_gene_similarity_matrix_by_batch(
                "GeneA", obo_path=self.obo_path, gaf_path=self.gaf_path  # type: ignore[arg-type]
            )

    def test_invalid_ontology_raises(self):
        """Unknown ontology value must raise ValueError."""
        opts = GeneSimilarityOptions(ontology="XX")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            compute_gene_similarity_matrix_by_batch(
                self.genes,
                obo_path=self.obo_path,
                gaf_path=self.gaf_path,
                go3_opts=opts,
            )

    def test_invalid_groupwise_raises(self):
        """Unknown groupwise value must raise ValueError."""
        opts = GeneSimilarityOptions(groupwise="bad")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            compute_gene_similarity_matrix_by_batch(
                self.genes,
                obo_path=self.obo_path,
                gaf_path=self.gaf_path,
                go3_opts=opts,
            )


if __name__ == "__main__":
    unittest.main()
