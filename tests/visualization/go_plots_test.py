"""
go_plots_test.py

Unit tests for GO plotting module.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from biocluster.visualization.go_plots import (
    GOPlotOptions,
    plot_gene_ratio,
    plot_go_metric,
    plot_qscore,
)


class TestGOPlots(unittest.TestCase):

    def setUp(self):
        """Create minimal valid dataset."""
        self.df = pd.DataFrame(
            {
                "name": ["t1", "t2", "t3"],
                "p_value": [0.01, 0.02, 0.05],
                "gene_ratio": [0.5, 0.3, 0.1],
                "intersection_size": [10, 5, 2],
                "qscore": [2.0, 1.7, 1.3],
            }
        )

    def test_basic_gene_ratio_plot(self):
        """Gene ratio plot should return figure."""
        fig = plot_gene_ratio(self.df, return_fig=True)
        self.assertIsNotNone(fig)

    def test_basic_qscore_plot(self):
        """Qscore plot should return figure."""
        fig = plot_qscore(self.df, return_fig=True)
        self.assertIsNotNone(fig)

    def test_html_saving(self):
        """HTML file should be written correctly, and cleaned up afterward."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = tmp_path / "plot.html"
            plot_gene_ratio(self.df, save_path=path)

            self.assertTrue(path.exists())

        self.assertFalse(tmp_path.exists())

    def test_invalid_metric(self):
        """Invalid metric should raise error."""
        with self.assertRaises(ValueError):
            plot_go_metric(self.df, metric="invalid")

    def test_missing_columns(self):
        """Missing required columns should raise."""
        df_bad = self.df.drop(columns=["p_value"])

        with self.assertRaises(ValueError):
            plot_gene_ratio(df_bad)

    def test_invalid_pvalue(self):
        """Negative p-values should raise."""
        df_bad = self.df.copy()
        df_bad["p_value"] = [-1, 0.1, 0.2]

        with self.assertRaises(ValueError):
            plot_gene_ratio(df_bad)

    def test_top_n_filter(self):
        """Top_n should reduce number of rows."""
        opts = GOPlotOptions(top_n=1)
        html = plot_gene_ratio(self.df, options=opts, return_html=True)

        self.assertTrue(isinstance(html, str))

    def test_return_html(self):
        """Return HTML should return string."""
        html = plot_gene_ratio(self.df, return_html=True)
        self.assertTrue(isinstance(html, str))

    def test_return_both(self):
        """Return both fig and html."""
        fig, html = plot_gene_ratio(self.df, return_fig=True, return_html=True)
        self.assertIsNotNone(fig)
        self.assertTrue(isinstance(html, str))

    ########################## Optional "source" column ##########################

    def test_gene_ratio_without_source_column_does_not_raise(self):
        """
        The documented schema (name, p_value, gene_ratio, intersection_size,
        qscore) has no "source" column; plotting must not require it.
        """
        self.assertNotIn("source", self.df.columns)

        fig = plot_gene_ratio(self.df, return_fig=True)

        self.assertNotIn("source", fig.data[0].hovertemplate)

    def test_qscore_without_source_column_does_not_raise(self):
        """Same guarantee as above, for the qscore (bar) plot."""
        self.assertNotIn("source", self.df.columns)

        fig = plot_qscore(self.df, return_fig=True)

        self.assertNotIn("source", fig.data[0].hovertemplate)

    def test_gene_ratio_includes_source_when_present(self):
        """When callers do provide a "source" column, it should still show up in the hover."""
        df_with_source = self.df.copy()
        df_with_source["source"] = ["GO:BP", "GO:MF", "GO:CC"]

        fig = plot_gene_ratio(df_with_source, return_fig=True)

        self.assertIn("source", fig.data[0].hovertemplate)

    def test_qscore_includes_source_when_present(self):
        """Same guarantee as above, for the qscore (bar) plot."""
        df_with_source = self.df.copy()
        df_with_source["source"] = ["GO:BP", "GO:MF", "GO:CC"]

        fig = plot_qscore(df_with_source, return_fig=True)

        self.assertIn("source", fig.data[0].hovertemplate)


if __name__ == "__main__":
    unittest.main()
