"""
Unit tests for GoEnrishment (refactored version).

Goals:
- Avoid real network calls (mock GProfiler).
- Validate input validation.
- Validate deterministic behavior.
- Validate scoring logic.
- Validate chunking logic in annotation.
- Validate wrappers compatibility.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from ParetoInsight_CPU.GoEnrishment import (
    go_enrichment,
    annotation_from_entrez_ids,
    GoEnrichment,
    AnnotationFromEntrezIDs,
    GoEnrichmentOptions,
    AnnotationOptions,
)


# ──────────────────────────────────────────────────────────────
# Helper: Fake GProfiler
# ──────────────────────────────────────────────────────────────

def fake_enrichment_df():
    return pd.DataFrame({
        "native": ["GO:0001", "GO:0002"],
        "p_value": [0.001, 0.01],
        "precision": [0.5, 0.25],
    })


def fake_annotation_df(block):
    return pd.DataFrame({
        "native": ["GO:0001", "GO:0002"],
        "intersections": [[block[0]], [block[-1]]],
    })


# ──────────────────────────────────────────────────────────────
# Test Suite
# ──────────────────────────────────────────────────────────────

class TestGoEnrichment(unittest.TestCase):

    def setUp(self):
        self.genes = ["1", "2", "3", "3"]  # duplicate included

    @patch("ParetoInsight_CPU.GoEnrishment.GProfiler")
    def test_go_enrichment_basic(self, mock_gp):
        """Ensure enrichment returns sorted DataFrame with qscore."""
        mock_instance = MagicMock()
        mock_instance.profile.return_value = fake_enrichment_df()
        mock_gp.return_value = mock_instance

        df = go_enrichment(self.genes)

        # duplicates removed
        mock_instance.profile.assert_called_once()
        self.assertEqual(len(df), 2)
        self.assertIn("gene_ratio", df.columns)
        self.assertIn("qscore", df.columns)

        # sorted by p_value ascending
        self.assertLessEqual(df["p_value"].iloc[0], df["p_value"].iloc[1])

    @patch("ParetoInsight_CPU.GoEnrishment.GProfiler")
    def test_empty_results(self, mock_gp):
        """Empty API response returns empty DataFrame."""
        mock_instance = MagicMock()
        mock_instance.profile.return_value = pd.DataFrame()
        mock_gp.return_value = mock_instance

        df = go_enrichment(["1", "2"])
        self.assertTrue(df.empty)

    def test_invalid_input(self):
        """Invalid inputs should raise."""
        with self.assertRaises(ValueError):
            go_enrichment([])

        with self.assertRaises(TypeError):
            go_enrichment(123)

    def test_invalid_threshold(self):
        """Threshold outside (0,1] should raise."""
        opts = GoEnrichmentOptions(user_threshold=1.5)
        with self.assertRaises(ValueError):
            go_enrichment(["1"], options=opts)

    @patch("ParetoInsight_CPU.GoEnrishment.GProfiler")
    def test_wrapper_compatibility(self, mock_gp):
        """Legacy wrapper should still work."""
        mock_instance = MagicMock()
        mock_instance.profile.return_value = fake_enrichment_df()
        mock_gp.return_value = mock_instance

        df = GoEnrichment(["1", "2"])
        self.assertFalse(df.empty)


class TestAnnotation(unittest.TestCase):

    def setUp(self):
        self.genes = ["A", "B", "C", "D"]

    @patch("ParetoInsight_CPU.GoEnrishment.GProfiler")
    def test_annotation_basic(self, mock_gp):
        """Annotation should return gene->terms dict."""
        mock_instance = MagicMock()

        def side_effect(**kwargs):
            block = kwargs["query"]
            return fake_annotation_df(block)

        mock_instance.profile.side_effect = side_effect
        mock_gp.return_value = mock_instance

        result = annotation_from_entrez_ids(self.genes)

        self.assertIsInstance(result, dict)
        self.assertTrue(len(result) > 0)

        for gene, terms in result.items():
            self.assertIsInstance(terms, list)

    @patch("ParetoInsight_CPU.GoEnrishment.GProfiler")
    def test_annotation_chunking(self, mock_gp):
        """Ensure chunking splits calls correctly."""
        mock_instance = MagicMock()

        calls = []

        def side_effect(**kwargs):
            calls.append(kwargs["query"])
            return fake_annotation_df(kwargs["query"])

        mock_instance.profile.side_effect = side_effect
        mock_gp.return_value = mock_instance

        opts = AnnotationOptions(chunk_size=2, n_threads=1)
        annotation_from_entrez_ids(self.genes, options=opts)

        # 4 genes / chunk_size=2 → 2 calls
        self.assertEqual(len(calls), 2)

    def test_annotation_invalid_input(self):
        """Invalid inputs should raise."""
        with self.assertRaises(ValueError):
            annotation_from_entrez_ids([])

        with self.assertRaises(TypeError):
            annotation_from_entrez_ids(123)

    @patch("ParetoInsight_CPU.GoEnrishment.GProfiler")
    def test_annotation_wrapper(self, mock_gp):
        """Legacy wrapper works."""
        mock_instance = MagicMock()

        def side_effect(**kwargs):
            return fake_annotation_df(kwargs["query"])

        mock_instance.profile.side_effect = side_effect
        mock_gp.return_value = mock_instance

        result = AnnotationFromEntrezIDs(["X", "Y"])
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()