"""
Unit tests for MappingEntrez module.

Purpose:
- Validate deterministic selection of minimum EntrezID.
- Validate correct fallback behavior (gProfiler → MyGene).
- Validate input validation logic.
- Ensure output preserves original order.
- Ensure NA handling works correctly.
- No real network calls (services are mocked).
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import logging

from ParetoInsight_CPU.MappingEntrez import (
    convert_to_entrez_id,
    MappingOptions,
)


class TestMappingEntrez(unittest.TestCase):

    ########################## Test Initialization ##########################

    def setUp(self):
        """Common reusable test data."""
        self.genes = ["A", "B", "C", "D"]
        self.options = MappingOptions(n_threads=1, chunk_size=2)
        
        # Disable logging during tests
        logging.disable(logging.CRITICAL)

    ########################## Input Validation ##########################

    def test_empty_input(self):
        """Purpose: empty list must raise ValueError."""
        with self.assertRaises(ValueError):
            convert_to_entrez_id([], self.options)

    def test_non_list_input(self):
        """Purpose: non-list input must raise ValueError."""
        with self.assertRaises(ValueError):
            convert_to_entrez_id("A", self.options)

    ########################## gProfiler Tests ##########################

    @patch("ParetoInsight_CPU.MappingEntrez.GProfiler")
    def test_gprofiler_basic_mapping(self, mock_gp):
        """
        Purpose:
        - Confirm gProfiler mapping works.
        - Confirm minimum EntrezID is selected.
        """

        # Mock dataframe returned by gProfiler
        df = pd.DataFrame({
            "incoming": ["A", "A", "B"],
            "converted": ["10", "5", "20"]
        })

        mock_instance = MagicMock()
        mock_instance.convert.return_value = df
        mock_gp.return_value = mock_instance

        result = convert_to_entrez_id(["A", "B"], self.options)

        # A → min(10,5) = 5
        self.assertEqual(result, ["5", "20"])

    @patch("ParetoInsight_CPU.MappingEntrez.GProfiler")
    def test_gprofiler_no_results(self, mock_gp):
        """Purpose: if gProfiler returns empty, fallback must handle."""
        mock_instance = MagicMock()
        mock_instance.convert.return_value = pd.DataFrame()
        mock_gp.return_value = mock_instance

        result = convert_to_entrez_id(["X"], self.options)

        self.assertEqual(result, ["NA"])

    ########################## MyGene Fallback ##########################

    @patch("ParetoInsight_CPU.MappingEntrez.GProfiler")
    @patch("ParetoInsight_CPU.MappingEntrez.mygene.MyGeneInfo")
    def test_mygene_fallback(self, mock_mg, mock_gp):
        """
        Purpose:
        - gProfiler returns empty.
        - MyGene resolves gene.
        """

        # gProfiler empty
        mock_gp.return_value.convert.return_value = pd.DataFrame()

        # MyGene returns valid mapping
        df = pd.DataFrame({
            "entrezgene": [100]
        }, index=["A"])

        df["notfound"] = False

        mock_mg.return_value.querymany.return_value = df

        result = convert_to_entrez_id(["A"], self.options)

        self.assertEqual(result, ["100"])

    @patch("ParetoInsight_CPU.MappingEntrez.GProfiler")
    @patch("ParetoInsight_CPU.MappingEntrez.mygene.MyGeneInfo")
    def test_mygene_multiple_entrez_selects_min(self, mock_mg, mock_gp):
        """
        Purpose:
        - MyGene returns multiple EntrezIDs.
        - Minimum must be selected deterministically.
        """

        mock_gp.return_value.convert.return_value = pd.DataFrame()

        df = pd.DataFrame({
            "entrezgene": [[50, 10, 40]]
        }, index=["A"])

        df["notfound"] = False

        mock_mg.return_value.querymany.return_value = df

        result = convert_to_entrez_id(["A"], self.options)

        self.assertEqual(result, ["10"])

    ########################## Order Preservation ##########################

    @patch("ParetoInsight_CPU.MappingEntrez.GProfiler")
    def test_output_order_preserved(self, mock_gp):
        """
        Purpose:
        - Output order must match input order.
        """

        df = pd.DataFrame({
            "incoming": ["C", "A"],
            "converted": ["3", "1"]
        })

        mock_gp.return_value.convert.return_value = df

        result = convert_to_entrez_id(["A", "B", "C"], self.options)

        # A → 1
        # B → NA
        # C → 3
        self.assertEqual(result, ["1", "NA", "3"])

    ########################## NA Handling ##########################

    @patch("ParetoInsight_CPU.MappingEntrez.GProfiler")
    def test_na_value_custom(self, mock_gp):
        """
        Purpose:
        - Custom NA value must be respected.
        """

        mock_gp.return_value.convert.return_value = pd.DataFrame()

        custom_opts = MappingOptions(na_value="MISSING")

        result = convert_to_entrez_id(["A"], custom_opts)

        self.assertEqual(result, ["MISSING"])


if __name__ == "__main__":
    unittest.main()