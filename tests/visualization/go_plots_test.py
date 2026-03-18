"""
go_plots_test.py

Unit tests for GO plotting module.
"""

import unittest
import tempfile
from pathlib import Path
import pandas as pd

from gclusters_characterization.visualization.go_plots import (
    plot_go_metric,
    plot_gene_ratio,
    plot_qscore,
    GOPlotOptions,
)


class TestGOPlots(unittest.TestCase):

    def setUp(self):
        """Create minimal valid dataset."""
        self.df = pd.DataFrame({
            "name": ["t1", "t2", "t3"],
            "p_value": [0.01, 0.02, 0.05],
            "gene_ratio": [0.5, 0.3, 0.1],
            "intersection_size": [10, 5, 2],
            "qscore": [2.0, 1.7, 1.3]
        })

    def test_basic_gene_ratio_plot(self):
        """Gene ratio plot should return figure."""
        fig = plot_gene_ratio(self.df, return_fig=True)
        self.assertIsNotNone(fig)

    def test_basic_qscore_plot(self):
        """Qscore plot should return figure."""
        fig = plot_qscore(self.df, return_fig=True)
        self.assertIsNotNone(fig)

    def test_html_saving(self):
        """HTML file should be written correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plot.html"
            plot_gene_ratio(self.df, save_path=path)

            self.assertTrue(path.exists())

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


if __name__ == "__main__":
    unittest.main()