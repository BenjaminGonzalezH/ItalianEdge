"""
cir_go_test.py

Unit tests for CirGO visualization module.

Functions tested:
1. plot_cirgo
2. _gene2terms_to_df
3. _prepare_cirgo_dataframe
"""

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from gclusters_characterization.visualization.cir_go import (
    plot_cirgo,
    _gene2terms_to_df,
    _prepare_cirgo_dataframe,
    CirGOOptions,
)


class TestGeneToDF(unittest.TestCase):

    def test_conversion_basic(self):
        """Gene-to-term mapping should be converted correctly."""
        data = {"g1": ["t1", "t2"], "g2": ["t2"]}

        df = _gene2terms_to_df(data)

        self.assertEqual(len(df), 3)
        self.assertIn("gene", df.columns)
        self.assertIn("go_term", df.columns)


class TestPrepareCirgoDF(unittest.TestCase):

    def test_filtering_and_sorting(self):
        """Terms should be filtered and sorted correctly."""
        df = pd.DataFrame({
            "gene": ["g1", "g2", "g3", "g1"],
            "go_term": ["t1", "t1", "t2", "t2"]
        })

        opts = CirGOOptions(min_genes_per_term=2, max_terms=1)

        result = _prepare_cirgo_dataframe(df, opts)

        self.assertEqual(len(result), 1)
        self.assertIn("gene_count", result.columns)
        self.assertIn("category", result.columns)


class TestPlotCirgo(unittest.TestCase):

    def setUp(self):
        """Create a simple valid dataset."""
        self.data = {
            "g1": ["t1", "t2"],
            "g2": ["t1"],
            "g3": ["t1"],
        }

    def test_basic_plot(self):
        """Plot should be created successfully."""
        fig = plot_cirgo(self.data)

        self.assertIsNotNone(fig)

    def test_save_html(self):
        """HTML file should be created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.html"

            plot_cirgo(self.data, save_html_to=path)

            self.assertTrue(path.exists())

    def test_invalid_input_type(self):
        """Non-dict input should raise error."""
        with self.assertRaises(TypeError):
            plot_cirgo(["bad_input"])

    def test_empty_input(self):
        """Empty input should raise error."""
        with self.assertRaises(ValueError):
            plot_cirgo({})

    def test_no_terms_after_filter(self):
        """Filtering removing all terms should raise error."""
        opts = CirGOOptions(min_genes_per_term=10)

        with self.assertRaises(ValueError):
            plot_cirgo(self.data, options=opts)


if __name__ == "__main__":
    unittest.main()