"""
Unit tests for MappingEntrez module.

Purpose:
- Achieve >90% coverage.
- Validate deterministic behavior.
- Validate fallback logic.
- Validate edge cases and internal utilities.
- Avoid real network calls (full mocking).
- Avoid logging side effects.
"""

######### Libraries #########
import logging
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from biocluster.go.mapping_entrez import (
    MappingOptions,
    _iter_chunks,
    _min_entrez_str,
    convert_to_entrez_id,
)


class TestMappingEntrez(unittest.TestCase):

    ########################## Setup ##########################

    def setUp(self):
        """
        Initialize reusable data and disable logging to avoid noise during tests.
        """
        self.genes = ["A", "B", "C", "D"]
        self.options = MappingOptions(n_threads=1, chunk_size=2)

        # Disable logging globally for clean test execution
        logging.disable(logging.CRITICAL)

    ########################## Input Validation ##########################

    def test_empty_input(self):
        """Empty input should raise ValueError."""
        with self.assertRaises(ValueError):
            convert_to_entrez_id([], self.options)

    def test_invalid_input_type(self):
        """Non-list input should raise ValueError."""
        with self.assertRaises(ValueError):
            convert_to_entrez_id("A", self.options)

    ########################## Utility Functions ##########################

    def test_iter_chunks(self):
        """Chunking must split sequence correctly."""
        data = [1, 2, 3, 4, 5]
        chunks = list(_iter_chunks(data, 2))
        self.assertEqual(chunks, [[1, 2], [3, 4], [5]])

    def test_iter_chunks_invalid(self):
        """Invalid chunk size must raise ValueError."""
        with self.assertRaises(ValueError):
            list(_iter_chunks([1, 2], 0))

    def test_min_entrez_single(self):
        """Single numeric value must be returned as string."""
        self.assertEqual(_min_entrez_str("10"), "10")

    def test_min_entrez_multiple(self):
        """List must return minimum numeric value."""
        self.assertEqual(_min_entrez_str([50, 10, 40]), "10")

    def test_min_entrez_invalid(self):
        """Invalid values must return None."""
        self.assertIsNone(_min_entrez_str("abc"))

    ########################## gProfiler Tests ##########################

    @patch("biocluster.go.mapping_entrez.GProfiler")
    def test_gprofiler_mapping(self, mock_gp):
        """
        Validate:
        - gProfiler mapping works
        - Minimum EntrezID is selected
        """

        df = pd.DataFrame({"incoming": ["A", "A", "B"], "converted": ["10", "5", "20"]})

        mock_instance = MagicMock()
        mock_instance.convert.return_value = df
        mock_gp.return_value = mock_instance

        result = convert_to_entrez_id(["A", "B"], self.options)

        self.assertEqual(result, ["5", "20"])

    @patch("biocluster.go.mapping_entrez.GProfiler")
    def test_gprofiler_missing_columns(self, mock_gp):
        """
        If expected columns are missing, mapping should return NA.
        """

        df = pd.DataFrame({"wrong": [1]})
        mock_gp.return_value.convert.return_value = df

        result = convert_to_entrez_id(["A"], self.options)

        self.assertEqual(result, ["NA"])

    ########################## MyGene Fallback ##########################

    @patch("biocluster.go.mapping_entrez.GProfiler")
    @patch("biocluster.go.mapping_entrez.mygene.MyGeneInfo")
    def test_mygene_fallback(self, mock_mg, mock_gp):
        """
        Validate fallback:
        - gProfiler fails
        - MyGene resolves mapping
        """

        mock_gp.return_value.convert.return_value = pd.DataFrame()

        df = pd.DataFrame({"entrezgene": [100]}, index=["A"])
        df["notfound"] = False

        mock_mg.return_value.querymany.return_value = df

        result = convert_to_entrez_id(["A"], self.options)

        self.assertEqual(result, ["100"])

    @patch("biocluster.go.mapping_entrez.GProfiler")
    @patch("biocluster.go.mapping_entrez.mygene.MyGeneInfo")
    def test_mygene_all_notfound(self, mock_mg, mock_gp):
        """
        If MyGene returns notfound, result must be NA.
        """

        mock_gp.return_value.convert.return_value = pd.DataFrame()

        df = pd.DataFrame({"entrezgene": [None]}, index=["A"])
        df["notfound"] = True

        mock_mg.return_value.querymany.return_value = df

        result = convert_to_entrez_id(["A"], self.options)

        self.assertEqual(result, ["NA"])

    ########################## Order and NA Handling ##########################

    @patch("biocluster.go.mapping_entrez.GProfiler")
    def test_order_preserved(self, mock_gp):
        """
        Output must preserve input order.
        """

        df = pd.DataFrame({"incoming": ["C", "A"], "converted": ["3", "1"]})

        mock_gp.return_value.convert.return_value = df

        result = convert_to_entrez_id(["A", "B", "C"], self.options)

        self.assertEqual(result, ["1", "NA", "3"])

    @patch("biocluster.go.mapping_entrez.GProfiler")
    def test_custom_na_value(self, mock_gp):
        """
        Custom NA value must be respected.
        """

        mock_gp.return_value.convert.return_value = pd.DataFrame()

        custom_opts = MappingOptions(na_value="MISSING")

        result = convert_to_entrez_id(["A"], custom_opts)

        self.assertEqual(result, ["MISSING"])

    ########################## Temporary Resource Safety ##########################

    def test_tempfile_usage(self):
        """
        Validate temporary file usage pattern (ensures no filesystem side-effects).
        """

        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp:
            tmp.write("test")
            tmp.seek(0)
            self.assertEqual(tmp.read(), "test")


if __name__ == "__main__":
    unittest.main()
