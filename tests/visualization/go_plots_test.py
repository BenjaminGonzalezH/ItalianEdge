"""
Unit tests for Go_Plots module.

Purpose:
- Validate correct GO plotting behavior.
- Validate DataFrame validation logic.
- Ensure HTML export works without leaving residual files.
- Confirm optional return modes (fig/html).
- Validate top_n filtering.
"""

# ──────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────
import unittest
import tempfile
import os
import logging
import numpy as np
import pandas as pd

from Graphs.Go_Plots import (
    plot_gene_ratio,
    plot_qscore,
    GOPlotOptions,
)


class TestGOPlots(unittest.TestCase):
    """Test suite for GO plotting logic and validation."""

    # ──────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────
    def setUp(self):
        """Create reusable small GO enrichment DataFrame."""
        logging.disable(logging.CRITICAL)

        self.df = pd.DataFrame({
            "name": ["GO:1", "GO:2", "GO:3", "GO:4"],
            "gene_ratio": [0.4, 0.3, 0.2, 0.1],
            "intersection_size": [20, 15, 10, 5],
            "p_value": [0.001, 0.01, 0.02, 0.05],
            "qscore": [5.0, 4.0, 3.0, 2.0]
        })

    # ──────────────────────────────────────────────────────────
    # Core Behavior Tests
    # ──────────────────────────────────────────────────────────

    def test_plot_gene_ratio_basic(self):
        """Confirm gene ratio plot generates HTML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gene_ratio.html")

            plot_gene_ratio(self.df, save_path=path)

            self.assertTrue(os.path.exists(path))

        self.assertFalse(os.path.exists(tmpdir))

    def test_plot_qscore_basic(self):
        """Confirm qscore plot generates HTML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "qscore.html")

            plot_qscore(self.df, save_path=path)

            self.assertTrue(os.path.exists(path))

        self.assertFalse(os.path.exists(tmpdir))

    # ──────────────────────────────────────────────────────────
    # Return Mode Tests
    # ──────────────────────────────────────────────────────────

    def test_return_figure(self):
        """Confirm return_fig=True returns Plotly Figure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fig.html")

            fig = plot_gene_ratio(
                self.df,
                save_path=path,
                return_fig=True
            )

            self.assertTrue(hasattr(fig, "to_html"))

    def test_return_html(self):
        """Confirm return_html=True returns HTML string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "html.html")

            html = plot_gene_ratio(
                self.df,
                save_path=path,
                return_html=True
            )

            self.assertIn("<div", html)

    def test_return_both(self):
        """Confirm return_fig=True and return_html=True returns tuple."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "both.html")

            fig, html = plot_gene_ratio(
                self.df,
                save_path=path,
                return_fig=True,
                return_html=True
            )

            self.assertTrue(hasattr(fig, "to_html"))
            self.assertIn("<div", html)

    # ──────────────────────────────────────────────────────────
    # Top-N Behavior
    # ──────────────────────────────────────────────────────────

    def test_top_n_filtering(self):
        """Confirm top_n limits number of plotted terms."""
        options = GOPlotOptions(top_n=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "topn.html")

            fig = plot_gene_ratio(
                self.df,
                save_path=path,
                options=options,
                return_fig=True
            )

            # Plotly stores data in fig.data[0]
            plotted_terms = len(fig.data[0].y)
            self.assertEqual(plotted_terms, 2)

    # ──────────────────────────────────────────────────────────
    # Validation Tests
    # ──────────────────────────────────────────────────────────

    def test_missing_column(self):
        """Missing required column should raise ValueError."""
        bad_df = self.df.drop(columns=["p_value"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.html")

            with self.assertRaises(ValueError):
                plot_gene_ratio(bad_df, save_path=path)

    def test_non_dataframe_input(self):
        """Non-DataFrame input should raise TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.html")

            with self.assertRaises(TypeError):
                plot_gene_ratio("not a df", save_path=path)

    def test_empty_dataframe(self):
        """Empty DataFrame should raise ValueError."""
        empty_df = pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.html")

            with self.assertRaises(ValueError):
                plot_gene_ratio(empty_df, save_path=path)

    def test_non_numeric_column(self):
        """Non-numeric p_value should raise TypeError."""
        bad_df = self.df.copy()
        bad_df["p_value"] = ["a", "b", "c", "d"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.html")

            with self.assertRaises(TypeError):
                plot_gene_ratio(bad_df, save_path=path)


if __name__ == "__main__":
    unittest.main()