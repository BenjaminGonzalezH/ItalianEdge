"""
go_enrishment_test.py

This module contains unit tests for GO enrichment and annotation utilities.

Functions tested:
1. go_enrichment:
   Validates enrichment computation, input handling, and scoring.

2. annotation_from_entrez_ids:
   Validates annotation retrieval and chunk processing.

3. _safe_neglog10:
   Validates numerical stability for logarithmic transformation.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from gclusters_characterization.go.go_enrishment import (
    go_enrichment,
    annotation_from_entrez_ids,
    GoEnrichmentOptions,
    AnnotationOptions,
    _safe_neglog10,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def fake_enrichment_df():
    """Create a minimal enrichment result for testing."""
    return pd.DataFrame({
        "native": ["GO:0001", "GO:0002"],
        "p_value": [0.001, 0.01],
        "precision": [0.5, 0.25],
    })


def fake_annotation_df(block):
    """Create a minimal annotation response for a gene block."""
    return pd.DataFrame({
        "native": ["GO:0001", "GO:0002"],
        "intersections": [[block[0]], [block[-1]]],
    })


# --------------------------------------------------
# Enrichment tests
# --------------------------------------------------

class TestGoEnrichment(unittest.TestCase):

    def setUp(self):
        """Prepare a small set of genes including duplicates and invalid values."""
        self.genes = ["1", "2", "3", "3", None, ""]

    @patch("gclusters_characterization.go.go_enrishment.GProfiler")
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

    @patch("gclusters_characterization.go.go_enrishment.GProfiler")
    def test_missing_precision_column(self, mock_gp):
        """Enrichment should still work without 'precision'."""
        df_mock = fake_enrichment_df().drop(columns=["precision"])

        mock_instance = MagicMock()
        mock_instance.profile.return_value = df_mock
        mock_gp.return_value = mock_instance

        df = go_enrichment(["1", "2"])

        self.assertIn("qscore", df.columns)

    @patch("gclusters_characterization.go.go_enrishment.GProfiler")
    def test_missing_pvalue_column(self, mock_gp):
        """If p_value is missing, qscore should not be computed."""
        df_mock = fake_enrichment_df().drop(columns=["p_value"])

        mock_instance = MagicMock()
        mock_instance.profile.return_value = df_mock
        mock_gp.return_value = mock_instance

        df = go_enrichment(["1", "2"])

        self.assertNotIn("qscore", df.columns)

    @patch("gclusters_characterization.go.go_enrishment.GProfiler")
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

    @patch("gclusters_characterization.go.go_enrishment.GProfiler")
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

    @patch("gclusters_characterization.go.go_enrishment.GProfiler")
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

    @patch("gclusters_characterization.go.go_enrishment.GProfiler")
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

    @patch("gclusters_characterization.go.go_enrishment.GProfiler")
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

    @patch("gclusters_characterization.go.go_enrishment.GProfiler")
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


if __name__ == "__main__":
    unittest.main()