"""
Unit tests for go_enrichment module (refactored version).

Goals
-----
- Avoid real network calls (mock GProfiler)
- Validate input normalization and deduplication
- Validate retry logic and robustness
- Validate enrichment scoring and sorting
- Validate annotation chunking and parallel behavior
- Achieve >90% coverage
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from biocluster.go.go_enrichment import (
    AnnotationOptions,
    GoEnrichmentOptions,
    _safe_neglog10,
    annotation_from_entrez_ids,
    go_enrichment,
)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def fake_enrichment_df():
    return pd.DataFrame(
        {
            "native": ["GO:0001", "GO:0002"],
            "p_value": [0.001, 0.01],
            "precision": [0.5, 0.25],
        }
    )


def fake_annotation_df(block):
    return pd.DataFrame(
        {
            "native": ["GO:0001", "GO:0002"],
            "intersections": [[block[0]], [block[-1]]],
        }
    )


# ─────────────────────────────────────────────
# Enrichment tests
# ─────────────────────────────────────────────


class TestGoEnrichment(unittest.TestCase):

    def setUp(self):
        self.genes = ["1", "2", "3", "3", None, ""]

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_basic_enrichment(self, mock_gp):
        """Basic enrichment returns sorted results with qscore."""
        mock_instance = MagicMock()
        mock_instance.profile.return_value = fake_enrichment_df()
        mock_gp.return_value = mock_instance

        df = go_enrichment(self.genes)

        # duplicates + cleaning applied
        self.assertEqual(mock_instance.profile.call_count, 1)

        self.assertIn("gene_ratio", df.columns)
        self.assertIn("qscore", df.columns)

        # sorted ascending
        self.assertTrue(df["p_value"].is_monotonic_increasing)

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_missing_precision_column(self, mock_gp):
        """Should still work without 'precision' column."""
        df_mock = fake_enrichment_df().drop(columns=["precision"])

        mock_instance = MagicMock()
        mock_instance.profile.return_value = df_mock
        mock_gp.return_value = mock_instance

        df = go_enrichment(["1", "2"])

        self.assertIn("qscore", df.columns)

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_missing_pvalue_column(self, mock_gp):
        """No qscore if p_value missing."""
        df_mock = fake_enrichment_df().drop(columns=["p_value"])

        mock_instance = MagicMock()
        mock_instance.profile.return_value = df_mock
        mock_gp.return_value = mock_instance

        df = go_enrichment(["1", "2"])

        self.assertNotIn("qscore", df.columns)

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_empty_response(self, mock_gp):
        """Empty API response returns empty DataFrame."""
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
        """Invalid threshold should raise."""
        opts = GoEnrichmentOptions(user_threshold=0)
        with self.assertRaises(ValueError):
            go_enrichment(["1"], options=opts)

    def test_invalid_sources(self):
        """Empty sources should raise."""
        opts = GoEnrichmentOptions(sources=())
        with self.assertRaises(ValueError):
            go_enrichment(["1"], options=opts)

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_retry_logic(self, mock_gp):
        """Ensure retry mechanism is triggered on failure."""
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


# ─────────────────────────────────────────────
# Annotation tests
# ─────────────────────────────────────────────


class TestAnnotation(unittest.TestCase):

    def setUp(self):
        self.genes = ["A", "B", "C", "D"]

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_basic_annotation(self, mock_gp):
        """Annotation returns gene→terms mapping."""
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
        """Chunking splits requests correctly."""
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
        """Failures in threads should not crash execution."""
        mock_instance = MagicMock()

        def side_effect(**kwargs):
            if len(kwargs["query"]) == 2:
                raise RuntimeError("fail")
            return fake_annotation_df(kwargs["query"])

        mock_instance.profile.side_effect = side_effect
        mock_gp.return_value = mock_instance

        result = annotation_from_entrez_ids(self.genes)

        self.assertIsInstance(result, dict)

    @patch("biocluster.go.go_enrichment.logger")
    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_thread_failure_emits_warning(self, mock_gp, mock_logger):
        """Every failed annotation chunk must emit a logger WARNING."""
        mock_instance = MagicMock()
        mock_instance.profile.side_effect = RuntimeError("network error")
        mock_gp.return_value = mock_instance

        # Use minimal retry/backoff and small chunks to keep the test fast.
        opts = AnnotationOptions(
            request_retries=1,
            backoff_base_seconds=0.0,
            chunk_size=2,
            n_threads=1,
        )

        result = annotation_from_entrez_ids(self.genes, options=opts)

        self.assertIsInstance(result, dict)
        self.assertTrue(
            mock_logger.warning.called,
            "logger.warning must be called when annotation chunks fail",
        )

    @patch("biocluster.go.go_enrichment.GProfiler")
    def test_missing_columns(self, mock_gp):
        """Missing expected columns returns empty dict."""
        mock_instance = MagicMock()
        mock_instance.profile.return_value = pd.DataFrame({"foo": [1]})
        mock_gp.return_value = mock_instance

        result = annotation_from_entrez_ids(self.genes)

        self.assertEqual(result, {})

    def test_invalid_inputs(self):
        """Invalid inputs should raise."""
        with self.assertRaises(ValueError):
            annotation_from_entrez_ids([])

        with self.assertRaises(TypeError):
            annotation_from_entrez_ids(123)


# ─────────────────────────────────────────────
# Utility tests
# ─────────────────────────────────────────────


class TestUtilities(unittest.TestCase):

    def test_safe_neglog10_normal(self):
        """Normal p-values should produce valid log scores."""
        val = _safe_neglog10(0.01)
        self.assertAlmostEqual(val, 2.0, places=5)

    def test_safe_neglog10_zero(self):
        """p=0 should not produce -inf."""
        val = _safe_neglog10(0)
        self.assertTrue(np.isfinite(val))

    def test_safe_neglog10_invalid(self):
        """Invalid values return NaN."""
        val = _safe_neglog10("bad")
        self.assertTrue(np.isnan(val))


if __name__ == "__main__":
    unittest.main()
